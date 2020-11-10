import matplotlib.pyplot as plt
import tifffile
import os
from os import listdir, makedirs, remove
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage.filters import uniform_filter
from skimage import io as skio
from skimage.measure import regionprops
from skimage.transform import resize, rescale
from skimage.filters import threshold_otsu, gaussian, threshold_triangle
from skimage.feature import peak_local_max
from skimage.color import label2rgb
from skimage.io import imsave
from skimage.segmentation import clear_border, watershed, find_boundaries
from skimage.morphology import (
    extrema, label, remove_small_objects, binary_erosion,
    disk, binary_dilation
)
from sklearn.cluster import KMeans
import cv2
import argparse
import copy
import datetime
from rowit import WindowView, crop_with_padding_mask
from joblib import Parallel, delayed, Memory
memory = Memory('./cachedir', verbose=0)


def imshowpair(A,B):
    plt.imshow(A,cmap='Purples')
    plt.imshow(B,cmap='Greens',alpha=0.5)
    plt.show()

    
def imshow(A):
    plt.imshow(A)
    plt.show()
    
def overlayOutline(outline,img):
    img2 = img.copy()
    stacked_img = np.stack((img2,)*3, axis=-1)
    stacked_img[outline > 0] = [65535, 0, 0]
    imshowpair(img2,stacked_img)
    
def normI(I):
    Irs=resize(I,(I.shape[0]//10,I.shape[1]//10) );
    p1 = np.percentile(Irs,10);
    J = I-p1;
    p99 = np.percentile(Irs,99.99);
    J = J/(p99-p1);
    return J

def contour_pm_watershed(
    contour_pm, sigma=2, h=0, tissue_mask=None,
    padding_mask=None, min_area=None, max_area=None
):
    if tissue_mask is None:
        tissue_mask = np.ones_like(contour_pm)
    padded = None
    if padding_mask is not None and np.any(padding_mask == 0):
        contour_pm, padded = crop_with_padding_mask(
            contour_pm, padding_mask, return_mask=True
        )
        tissue_mask = crop_with_padding_mask(
            tissue_mask, padding_mask
        )
    
    maxima = peak_local_max(
        extrema.h_maxima(
            ndi.gaussian_filter(np.invert(contour_pm), sigma=sigma),
            h=h
        ),
        indices=False,
        footprint=np.ones((3, 3))
    )
    maxima = label(maxima).astype(np.int32)
    
    # Passing mask into the watershed function will exclude seeds outside
    # of the mask, which gives fewer and more accurate segments
    maxima = watershed(
        contour_pm, maxima, watershed_line=True, mask=tissue_mask
    ) > 0
    
    if min_area is not None and max_area is not None:
        maxima = label(maxima, connectivity=1).astype(np.int32)
        areas = np.bincount(maxima.ravel())
        size_passed = np.arange(areas.size)[
            np.logical_and(areas > min_area, areas < max_area)
        ]
        maxima *= np.isin(maxima, size_passed)
        np.greater(maxima, 0, out=maxima)

    if padded is None:
        return maxima.astype(np.bool)
    else:
        padded[padded == 1] = maxima.flatten()
        return padded.astype(np.bool)

def S3NucleiSegmentationWatershed(nucleiPM,nucleiImage,logSigma,TMAmask,nucleiFilter,nucleiRegion):
    nucleiContours = nucleiPM[:,:,1]
    nucleiCenters = nucleiPM[:,:,0]
    del nucleiPM
    mask = resize(TMAmask,(nucleiImage.shape[0],nucleiImage.shape[1]),order = 0)>0
 
    if len(logSigma)==1:
         nucleiDiameter  = [logSigma*0.5, logSigma*1.5]
    else:
         nucleiDiameter = logSigma
    
    win_view_setting = WindowView(nucleiContours.shape, 2000, 500)

    nucleiContours = win_view_setting.window_view_list(nucleiContours)
    padding_mask = win_view_setting.padding_mask()
    mask = win_view_setting.window_view_list(mask)

    maxArea = (logSigma[1]**2)*3/4
    minArea = (logSigma[0]**2)*3/4

    foregroundMask = np.array(
        Parallel(n_jobs=10)(delayed(contour_pm_watershed)(
            img, sigma=logSigma[1]/30, h=logSigma[1]/30, tissue_mask=tm,
            padding_mask=m, min_area=minArea, max_area=maxArea
        ) for img, tm, m in zip(nucleiContours, mask, padding_mask))
    )

    del nucleiContours, mask, padding_mask

    foregroundMask = win_view_setting.reconstruct(foregroundMask)

    if nucleiFilter == 'IntPM':
        int_img = nucleiCenters
    elif nucleiFilter == 'Int':
        int_img = nucleiImage
    elif nucleiFilter == 'LoG':
        int_img = np.log1p(nucleiImage)
    
    return foregroundMask, int_img, minArea, maxArea

def filter_nuclei_mask(foregroundMask, int_img, minArea, maxArea, minSolidity=0.7):
    wv_setting = WindowView(foregroundMask.shape, 20000, 1000)
    foregroundMask = wv_setting.window_view_list(foregroundMask)
    int_img = wv_setting.window_view_list(int_img)

    def wrapper(foregroundMask, int_img):
        print('    ', datetime.datetime.now(), 'label')
        foregroundMask = label(foregroundMask, connectivity=1)

        print('    ', datetime.datetime.now(), 'regionprops')
        P = regionprops(foregroundMask, int_img)

        def props_of_keys(prop, keys):
            return [prop[k] for k in keys]

        prop_keys = ['mean_intensity', 'area', 'solidity', 'label']
        try:
            mean_ints, areas, solidities, labels = np.array(
                Parallel(n_jobs=8)(delayed(props_of_keys)(prop, prop_keys) 
                    for prop in P
                )
            ).T
            del P
        except:
            return np.zeros_like(foregroundMask).astype(np.bool)

        try:
            MITh = threshold_otsu(mean_ints)
        except ValueError:
            MITh = 0
        
        passed = np.logical_and.reduce((
            np.greater(mean_ints, MITh),
            np.logical_and(areas > minArea, areas < maxArea),
            np.greater(solidities, minSolidity)
        ))

        # set failed mask label to zero
        foregroundMask *= np.isin(foregroundMask, labels[passed])

        np.greater(foregroundMask, 0, out=foregroundMask)
        foregroundMask = label(foregroundMask, connectivity=1).astype(np.int32)

        return foregroundMask > 0

    foregroundMask = [
        wrapper(m, i)
        for m, i in zip(foregroundMask, int_img)
    ]
    print('    ', datetime.datetime.now(), 'reconstruct filtered')
    foregroundMask = wv_setting.reconstruct(np.array(foregroundMask))
    return foregroundMask

def bwmorph(mask,radius):
    mask = np.array(mask,dtype=np.uint8)
    #labels = label(mask)
    background = nucleiMask == 0
    distances, (i, j) = ndi.distance_transform_edt(background, return_indices=True)
    cellMask = nucleiMask.copy()
    finalmask = background & (distances <= radius)
    cellMask[finalmask] = nucleiMask[i[finalmask], j[finalmask]]

#    imshowpair(cellMask,mask)
    return cellMask
#    imshow(fg)
#    fg = cv2.dilate(mask,ndimage.generate_binary_structure(2, 2))
#    bg = 1-fg-mask
#    imshowpair(bg,mask)

def S3CytoplasmSegmentation(nucleiMask,cyto,mask,cytoMethod='distanceTransform',radius = 5):
    
    if cytoMethod == 'distanceTransform':
        mask = (nucleiMask + resize(mask,(nucleiMask.shape[0],nucleiMask.shape[1]),order=0))>0
        mask = np.array(mask,dtype=np.uint32)
        gdist = ndi.distance_transform_edt(1-(nucleiMask>0))
        markers= nucleiMask
    elif cytoMethod == 'hybrid':
        cytoBlur = gaussian(cyto,2)
        del cyto
        c1 = uniform_filter(cytoBlur, 3, mode='reflect')
        c2 = uniform_filter(cytoBlur*cytoBlur, 3, mode='reflect')
        del cytoBlur
        grad = np.sqrt(c2 - c1*c1)*np.sqrt(9./8)
        del c1, c2
        grad[np.isnan(grad)]=0
        gdist= np.sqrt(np.square(grad) + 0.000001*np.amax(grad)/np.amax(gdist)*np.square(gdist))
        del grad
        bg = binary_erosion(np.invert(mask),disk(radius, np.uint8))
        markers=nucleiMask.copy()
        markers[bg==1] = np.amax(nucleiMask)+1
        del bg
        markers = label(markers>0,connectivity=1)
        mask = np.ones(nucleiMask.shape)
    elif cytoMethod == 'ring':
        mask *= binary_dilation(nucleiMask > 0, selem=disk(radius))
        markers = nucleiMask
        gdist = -markers
    
    # With the rolling window approach, the ring and hybrid give the 
    # same results as the whole image approach. The distance-transform
    # approach gives quite different results which requires more 
    # investigation.
    # Update - the differences are gone if set watershed_line=False

    # settings for window operation
    win_view_setting = WindowView(nucleiMask.shape, 2000, 500)
    
    print('    ', datetime.datetime.now(), 'watershed')
    gdist = win_view_setting.window_view_list(gdist)
    markers = win_view_setting.window_view_list(markers)
    mask = win_view_setting.window_view_list(mask)

    # Wrapper to reduce memory useage
    def watershed_return_binary(img, marker, mask):
        return (
            watershed(img, marker, mask=mask, watershed_line=True) > 0
        ).astype(np.bool)
    cellMask = np.array(
        Parallel(n_jobs=8)(delayed(watershed_return_binary)(
            g, m, w_m
        ) for g, m, w_m in zip(gdist, markers, mask))
    )
    del gdist, markers, mask

    cellMask = win_view_setting.reconstruct(cellMask)
    print('    ', datetime.datetime.now(), 'label')
    cellMask = chop_label(cellMask)
    cellMask = clear_border(cellMask)
    passed = np.unique(
        np.multiply(cellMask, nucleiMask > 0)
    )
    cellMask *= np.isin(cellMask, passed)
    cellMask = chop_label(cellMask > 0).astype(np.int32)
    # Passing the out kwarg into numpy ufuncs will change the target
    # variable in-place, reassigning does not have this effect
    nucleiMask = np.multiply(nucleiMask > 0, cellMask)
    cytoplasmMask = np.subtract(cellMask, nucleiMask)
    
    return cytoplasmMask, nucleiMask, cellMask
    
def exportMasks(mask,image,outputPath,filePrefix,fileName,saveFig=True,saveMasks = True):
    outputPath =outputPath + os.path.sep + filePrefix
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    if saveMasks ==True:
        kwargs={}
        kwargs['bigtiff'] = True
        kwargs['photometric'] = 'minisblack'
        resolution = np.round(1)
        kwargs['resolution'] = (resolution, resolution, 'cm')
        kwargs['metadata'] = None
        imsave(outputPath + os.path.sep + fileName + 'Mask.tif',mask, plugin="tifffile", check_contrast=False)
        
    if saveFig== True:
        mask=np.uint8(mask>0)
        edges=find_boundaries(mask, mode='outer')
        stacked_img=np.stack((np.uint16(edges)*65535,image),axis=0)
        tifffile.imsave(outputPath + os.path.sep + fileName + 'Outlines.tif',stacked_img)
        
def auto_coarse_mask(nucleiContours):
    thumb_nucleiContours = rescale(nucleiContours, 1/50)
    mask_threshold = threshold_triangle(thumb_nucleiContours)
    tissue_mask = resize(
        thumb_nucleiContours>mask_threshold, 
        nucleiContours.shape
    ).astype(np.uint8)
    return tissue_mask
        
    # assign nan to tissue mask

def normalize_img_channel(img):
    if img.ndim == 2:
        return img.reshape(1, *img.shape)
    elif img.ndim == 3:
        if 3 in img.shape:
            channel_idx = img.shape.index(3)
            return np.moveaxis(img, channel_idx, 0)
        else:
            return img
    else:
        raise NotImplementedError(
            'image of shape {} is not supported'.format(img.shape)
        )

def chop_label(mask):
    height, _ = mask.shape
    chop_height = 10000
    
    mask = mask.astype(np.int32)
    id_max = 0
    for i in range(np.ceil(height / chop_height).astype(int)):
        r_s, r_e = i*chop_height, (i+1)*chop_height
        labeled, id_max = label(mask[r_s:r_e, :] > 0, connectivity=1, return_num=True)
        labeled[labeled != 0] += id_max
        id_max += id_max
        mask[r_s:r_e, :] = labeled
    
    return mask

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--imagePath")
    parser.add_argument("--contoursClassProbPath")
    parser.add_argument("--nucleiClassProbPath")
    parser.add_argument("--outputPath")
    parser.add_argument("--dearrayPath")
    parser.add_argument("--maskPath")
    parser.add_argument("--probMapChan",type = int, default = -1)
    parser.add_argument("--mask",choices=['TMA', 'tissue','none'],default = 'tissue')
    parser.add_argument("--crop",choices=['interactiveCrop','autoCrop','noCrop','dearray','plate'], default = 'noCrop')
    parser.add_argument("--cytoMethod",choices=['hybrid','distanceTransform','bwdistanceTransform','ring'],default = 'distanceTransform')
    parser.add_argument("--nucleiFilter",choices=['IntPM','LoG','Int','none'],default = 'IntPM')
    parser.add_argument("--nucleiRegion",choices=['watershedContourDist','watershedContourInt','watershedBWDist','dilation'], default = 'watershedContourInt')
    parser.add_argument("--segmentCytoplasm",choices = ['segmentCytoplasm','ignoreCytoplasm'],default = 'segmentCytoplasm')
    parser.add_argument("--cytoDilation",type = int, default = 5)
    parser.add_argument("--logSigma",type = int, nargs = '+', default = [3, 60])
    parser.add_argument("--CytoMaskChan",type=int, nargs = '+', default=[1])
    parser.add_argument("--TissueMaskChan",type=int, nargs = '+', default=-1)
    parser.add_argument("--saveMask",action='store_false')
    parser.add_argument("--saveFig",action='store_false')
    args = parser.parse_args()
    
    # gather filename information
    #exemplar001
#    imagePath = 'D:/LSP/cycif/testsets/exemplar-001/registration/exemplar-001.ome.tif'
#    outputPath = 'D:/LSP/cycif/testsets/exemplar-001/segmentation'
#    nucleiClassProbPath = 'D:/LSP/cycif/testsets/exemplar-001/prob_maps/exemplar-001_NucleiPM_25.tif'
#    contoursClassProbPath = 'D:/LSP/cycif/testsets/exemplar-001/prob_maps/exemplar-001_ContoursPM_25.tif'
#    maskPath = 'D:/LSP/cycif/testsets/exemplar-001/dearray/masks/A1_mask.tif'
#    args.cytoMethod = 'hybrid'
	
	#plate 
#    imagePath = 'Y:/sorger/data/computation/Jeremy/caitlin-ddd-cycif-registered/Plate1/E3_fld_1/registration/E3_fld_1.ome.tif'
#    outputPath = 'Y:/sorger/data/computation/Jeremy/caitlin-ddd-cycif-registered/Plate1/E3_fld_1/segmentation'
#    nucleiClassProbPath = 'Y:/sorger/data/computation/Jeremy/caitlin-ddd-cycif-registered/Plate1/E3_fld_1/prob_maps/E3_fld_1_NucleiPM_1.tif'
#    contoursClassProbPath = 'Y:/sorger/data/computation/Jeremy/caitlin-ddd-cycif-registered/Plate1/E3_fld_1/prob_maps/E3_fld_1_ContoursPM_1.tif'
#    maskPath = 'D:/LSP/cycif/testsets/exemplar-001/dearray/masks/A1_mask.tif'
#    args.crop = 'plate'
#    args.cytoMethod ='hybrid'
        
    #large tissue
#    imagePath =  'Y:/sorger/data/RareCyte/Connor/Z155_PTCL/Ton_192/registration/Ton_192.ome.tif'
#    outputPath = 'D:/LSP/cycif/testsets/exemplar-001/segmentation'
#    nucleiClassProbPath = 'Y:/sorger/data/RareCyte/Connor/Z155_PTCL/Ton_192/prob_maps/Ton_192_NucleiPM_41.tif'
#    contoursClassProbPath = 'Y:/sorger/data/RareCyte/Connor/Z155_PTCL/Ton_192/prob_maps/Ton_192_ContoursPM_41.tif'
#    maskPath = 'D:/LSP/cycif/testsets/exemplar-001/dearray/masks/A1_mask.tif'
    
    imagePath = args.imagePath
    outputPath = args.outputPath
    nucleiClassProbPath = args.nucleiClassProbPath
    contoursClassProbPath = args.contoursClassProbPath
    maskPath = args.maskPath
       
    fileName = os.path.basename(imagePath)
    filePrefix = fileName[0:fileName.index('.')]
    
    # get channel used for nuclei segmentation
    if args.probMapChan==-1:
        test = os.path.basename(contoursClassProbPath)
        nucMaskChan = int(test.split('ContoursPM_')[1].split('.')[0])-1
        
    else:
        nucMaskChan = args.probMapChan
        
    if args.TissueMaskChan==-1:
        TissueMaskChan = copy.copy(args.CytoMaskChan)
        TissueMaskChan.append(nucMaskChan)
    else:
        TissueMaskChan = args.TissueMaskChan[:]
        TissueMaskChan.append(nucMaskChan)
            
    #crop images if needed
    print(datetime.datetime.now(), 'Cropping image')
    if args.crop == 'interactiveCrop':
        nucleiCrop = tifffile.imread(imagePath,key = nucMaskChan)
        r=cv2.selectROI(resize(nucleiCrop,(nucleiCrop.shape[0] // 10, nucleiCrop.shape[1] // 10)))
        cv2.destroyWindow('select')
        rect=np.transpose(r)*10
        PMrect= [rect[1], rect[0], rect[3], rect[2]]
        nucleiCrop = nucleiCrop[int(rect[1]):int(rect[1]+rect[3]), int(rect[0]):int(rect[0]+rect[2])]
    elif args.crop == 'noCrop' or args.crop == 'dearray'  or args.crop == 'plate':
        nucleiCrop = tifffile.imread(imagePath,key = nucMaskChan)
        rect = [0, 0, nucleiCrop.shape[0], nucleiCrop.shape[1]]
        PMrect= rect
    nucleiProbMaps = tifffile.imread(nucleiClassProbPath)
    nucleiProbMaps = normalize_img_channel(nucleiProbMaps)[0]
    nucleiPM = nucleiProbMaps[int(PMrect[0]):int(PMrect[0]+PMrect[2]), int(PMrect[1]):int(PMrect[1]+PMrect[3])]
    nucleiProbMaps = tifffile.imread(contoursClassProbPath)
    nucleiProbMaps = normalize_img_channel(nucleiProbMaps)
    nucleiProbMaps = nucleiProbMaps[1] if nucleiProbMaps.shape[0] == 3 else nucleiProbMaps[0]
    PMSize = nucleiProbMaps.shape
    nucleiPM = np.dstack((nucleiPM,nucleiProbMaps[int(PMrect[0]):int(PMrect[0]+PMrect[2]), int(PMrect[1]):int(PMrect[1]+PMrect[3])]))

    # mask the core/tissue
    print(datetime.datetime.now(), 'Computing tissue mask')
    if args.crop == 'dearray':
        try:
            TMAmask = tifffile.imread(maskPath)
        except ValueError:
            auto_coarse_mask = memory.cache(auto_coarse_mask)
            TMAmask = auto_coarse_mask(nucleiPM[..., 1])
        
    elif args.crop =='plate':
        TMAmask = np.ones(nucleiCrop.shape)
		
    else:
        tissue = np.empty((len(TissueMaskChan),nucleiCrop.shape[0],nucleiCrop.shape[1]),dtype=np.uint16)
        count=0
        if args.crop == 'noCrop':
            for iChan in TissueMaskChan:
                tissueCrop =tifffile.imread(imagePath,key=iChan)
                tissue_gauss = gaussian(tissueCrop,1)
                #tissue_gauss[tissue_gauss==0]=np.nan
                tissue[count,:,:] =np.log2(tissue_gauss+1)>threshold_otsu(np.log2(tissue_gauss+1))
                count+=1
        else:
            for iChan in TissueMaskChan:
                tissueCrop = tifffile.imread(imagePath,key=iChan)
                tissue_gauss = gaussian(tissueCrop[int(PMrect[0]):int(PMrect[0]+PMrect[2]), int(PMrect[1]):int(PMrect[1]+PMrect[3])],1)
                tissue[count,:,:] =  np.log2(tissue_gauss+1)>threshold_otsu(np.log2(tissue_gauss+1))
                count+=1
        TMAmask = np.max(tissue,axis = 0)

 #       tissue_gauss = tissueCrop
#        tissue_gauss1 = tissue_gauss.astype(float)
#        tissue_gauss1[tissue_gauss>np.percentile(tissue_gauss,99)]=np.nan
#        TMAmask = np.log2(tissue_gauss+1)>threshold_otsu(np.log2(tissue_gauss+1))
        #imshow(TMAmask)
        del tissue_gauss, tissue

    # nuclei segmentation
    print(datetime.datetime.now(), 'Segmenting nuclei')
    nucleiMask, int_img, minArea, maxArea = S3NucleiSegmentationWatershed(
        nucleiPM,nucleiCrop,args.logSigma,TMAmask,args.nucleiFilter,args.nucleiRegion
    )
    filter_nuclei_mask = memory.cache(filter_nuclei_mask)
    nucleiMask = filter_nuclei_mask(
        nucleiMask, int_img, minArea, maxArea, minSolidity=0.7
    )

    print(datetime.datetime.now(), 'Label nuclei mask')
    nucleiMask = chop_label(nucleiMask)

    del nucleiPM
    # cytoplasm segmentation
    print(datetime.datetime.now(), 'Segmenting cytoplasm')
    if args.segmentCytoplasm == 'segmentCytoplasm':
        count =0
        if args.cytoMethod != 'ring':
            if args.crop == 'noCrop' or args.crop == 'dearray' or args.crop == 'plate':
                cyto=np.empty((len(args.CytoMaskChan),nucleiCrop.shape[0],nucleiCrop.shape[1]),dtype=np.uint16)    
                for iChan in args.CytoMaskChan:
                    cyto[count,:,:] =  skio.imread(imagePath, key=iChan)
                    count+=1
            else:
                cyto=np.empty((len(args.CytoMaskChan),rect[3],rect[2]),dtype=np.int16)
                for iChan in args.CytoMaskChan:
                    cytoFull= skio.imread(imagePath, key=iChan)
                    cyto[count,:,:] = cytoFull[int(PMrect[0]):int(PMrect[0]+PMrect[2]), int(PMrect[1]):int(PMrect[1]+PMrect[3])]
                    count+=1
        
            cyto = np.amax(cyto,axis=0)
            cytoplasmMask,nucleiMaskTemp,cellMask = S3CytoplasmSegmentation(nucleiMask,cyto,TMAmask,args.cytoMethod,args.cytoDilation)
            exportMasks(nucleiMaskTemp,nucleiCrop,outputPath,filePrefix,'nuclei',args.saveFig,args.saveMask)
            exportMasks(cytoplasmMask,cyto,outputPath,filePrefix,'cyto',args.saveFig,args.saveMask)
            exportMasks(cellMask,cyto,outputPath,filePrefix,'cell',args.saveFig,args.saveMask)

        print(datetime.datetime.now(), '  Segmenting cytoplasm - ring')
        cytoplasmMask,nucleiMaskTemp,cellMask = S3CytoplasmSegmentation(nucleiMask,None,TMAmask,'ring',args.cytoDilation)
        exportMasks(nucleiMaskTemp,nucleiCrop,outputPath,filePrefix,'nucleiRing',args.saveFig,args.saveMask)
        exportMasks(cytoplasmMask,nucleiCrop,outputPath,filePrefix,'cytoRing',args.saveFig,args.saveMask)
        exportMasks(cellMask,nucleiCrop,outputPath,filePrefix,'cellRing',args.saveFig,args.saveMask)
        
    elif args.segmentCytoplasm == 'ignoreCytoplasm':
        exportMasks(nucleiMask,nucleiCrop,outputPath,filePrefix,'nuclei')
        cellMask = nucleiMask
        exportMasks(nucleiMask,nucleiCrop,outputPath,filePrefix,'cell')
        cytoplasmMask = nucleiMask
        
        #fix bwdistance watershed
   
                