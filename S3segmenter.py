import matplotlib.pyplot as plt
import tifffile
import os
import numpy as np
from skimage import io as skio
# from scipy.ndimage import *
import scipy.ndimage as ndi
from skimage.measure import regionprops
from skimage.transform import resize
from skimage.filters import threshold_otsu, gaussian
from skimage.feature import peak_local_max
from skimage.color import label2rgb
from skimage.io import imsave
from skimage.segmentation import clear_border, watershed
from skimage.morphology import (
    extrema, label, remove_small_objects, binary_erosion,
    disk, 
)
from scipy.ndimage.filters import uniform_filter
from os.path import *
from os import listdir, makedirs, remove
from sklearn.cluster import KMeans
import pickle
import shutil
import fnmatch
import cv2
import sys
import argparse
import re
import copy
import datetime
from skimage.util import view_as_windows, montage
from joblib import Parallel, delayed


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

def view_as_windows_overlap(
    img, block_size, overlap_size
): 
    init_shape = img.shape
    step_size = block_size - overlap_size

    padded_shape = np.array(init_shape) + overlap_size
    n = np.ceil((padded_shape - block_size) / step_size)
    padded_shape = (block_size + (n * step_size)).astype(np.int)

    half = int(overlap_size / 2)
    img = np.pad(img, (
        (half, padded_shape[0] - init_shape[0] - half), 
        (half, padded_shape[1] - init_shape[1] - half),
    ), mode='edge')

    return view_as_windows(img, block_size, step_size)

def reconstruct_from_windows(
    window_view, block_size, overlap_size, out_shape=None
):
    grid_shape = window_view.shape[:2]

    start = int(overlap_size / 2)
    end = int(block_size - start)

    window_view = window_view.reshape(
        (-1, block_size, block_size)
    )[..., start:end, start:end]

    if out_shape:
        re, ce = out_shape
    else:
        re, ce = None, None

    return montage(
        window_view, grid_shape=grid_shape, 
    )[:re, :ce]

def local_max_in_gaussian(img, sigma, h):
    return peak_local_max(
        extrema.h_maxima(
            ndi.gaussian_filter(np.invert(img), sigma=sigma),
            h=h
        ),
        indices=False,
        footprint=np.ones((3, 3))
    )

def S3NucleiSegmentationWatershed(nucleiPM,nucleiImage,logSigma,TMAmask,nucleiFilter,nucleiRegion):
    nucleiContours = nucleiPM[:,:,1]
    nucleiCenters = nucleiPM[:,:,0]
    del nucleiPM
    mask = resize(TMAmask,(nucleiImage.shape[0],nucleiImage.shape[1]),order = 0)>0
 
    if len(logSigma)==1:
         nucleiDiameter  = [logSigma*0.5, logSigma*1.5]
    else:
         nucleiDiameter = logSigma
    logMask = nucleiCenters > 150
    # dist_trans_img = ndi.distance_transform_edt(logMask)
    
    img_shape = nucleiContours.shape
    block_size = 2000
    overlap_size = 500
    window_view_shape = view_as_windows_overlap(
        nucleiContours, block_size, overlap_size
    ).shape

    nucleiContours = view_as_windows_overlap(
        nucleiContours, block_size, overlap_size
    ).reshape(-1, block_size, block_size)

    print('    ', datetime.datetime.now(), 'local max')
    fgm = np.array(
        Parallel(n_jobs=6)(delayed(local_max_in_gaussian)(
            img, logSigma[1]/30, logSigma[1]/30
        ) for img in nucleiContours)
    )
    fgm = reconstruct_from_windows(
        fgm.reshape(window_view_shape),
        block_size, overlap_size, img_shape
    )
    fgm = label(fgm).astype(np.int32)

    print('    ', datetime.datetime.now(), 'watershed')
    fgm = view_as_windows_overlap(
        fgm, block_size, overlap_size
    ).reshape(-1, block_size, block_size)

    foregroundMask = np.array(
        Parallel(n_jobs=6)(delayed(watershed)(
            n, f, watershed_line=True
        ) for n, f in zip(nucleiContours, fgm))
    ) > 0
    
    del fgm, nucleiContours

    foregroundMask = reconstruct_from_windows(
        foregroundMask.reshape(window_view_shape),
        block_size, overlap_size, img_shape
    )
    foregroundMask *= mask
    remove_small_objects(
        foregroundMask, 
        min_size=np.floor((logSigma[0]**2)*3/4), 
        in_place=True
    )
    foregroundMask = label(foregroundMask, connectivity=1).astype(np.int32)

    if nucleiFilter == 'IntPM':
        int_img = nucleiCenters
    elif nucleiFilter == 'Int':
        int_img = nucleiImage

    print('    ', datetime.datetime.now(), 'regionprops')
    P = regionprops(foregroundMask, int_img)

    def props_of_keys(prop, keys):
        return [prop[k] for k in keys]

    prop_keys = ['mean_intensity', 'area', 'solidity', 'label']
    mean_ints, areas, solidities, labels = np.array(
        Parallel(n_jobs=6)(delayed(props_of_keys)(prop, prop_keys) 
            for prop in P
        )
    ).T
    del P

    # kmeans = KMeans(n_clusters=2).fit(mean_int.reshape(-1,1))
    MITh = threshold_otsu(mean_ints)

    maxArea = (logSigma[1]**2)*3/4
    minArea = (logSigma[0]**2)*3/4
    minSolidity = 0.8

    passed = np.logical_and.reduce((
        np.greater(mean_ints, MITh),
        np.logical_and(areas > minArea, areas < maxArea),
        np.greater(solidities, minSolidity)
    ))

    # set failed mask label to zero
    foregroundMask *= np.isin(foregroundMask, labels[passed])

    np.greater(foregroundMask, 0, out=foregroundMask)
    foregroundMask = label(foregroundMask, connectivity=1).astype(np.int32)

    return foregroundMask
    
#    img2 = nucleiImage.copy()
#    stacked_img = np.stack((img2,)*3, axis=-1)
#    stacked_img[X > 0] = [65535, 0, 0]
#    imshowpair(img2,stacked_img)

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
    mask = (nucleiMask + resize(mask,(nucleiMask.shape[0],nucleiMask.shape[1]),order=0))>0
    gdist = ndi.distance_transform_edt(1-(nucleiMask>0))
    if cytoMethod == 'distanceTransform':
        mask = np.array(mask,dtype=np.uint32)
        markers= nucleiMask
    elif cytoMethod == 'hybrid':
        cytoBlur = gaussian(cyto,2)
        c1 = uniform_filter(cytoBlur, 3, mode='reflect')
        c2 = uniform_filter(cytoBlur*cytoBlur, 3, mode='reflect')
        grad = np.sqrt(c2 - c1*c1)*np.sqrt(9./8)
        grad[np.isnan(grad)]=0
        gdist= np.sqrt(np.square(grad) + 0.000001*np.amax(grad)/np.amax(gdist)*np.square(gdist))
        bg = binary_erosion(np.invert(mask),disk(radius, np.uint8))
        markers=nucleiMask.copy()
        markers[bg==1] = np.amax(nucleiMask)+1
        markers = label(markers>0,connectivity=1)
        mask = np.ones(nucleiMask.shape)
        del bg
    elif cytoMethod == 'ring':
        mask =np.array(bwmorph(nucleiMask,radius)*mask,dtype=np.uint32)>0
        markers= nucleiMask
    
    print('    ', datetime.datetime.now(), 'watershed')
    cellMask  =clear_border(watershed(gdist,markers,watershed_line=True))
    del gdist, markers, cyto
    cellMask = np.array(cellMask*mask,dtype=np.uint32)
	
    finalCellMask = np.zeros(cellMask.shape,dtype=np.uint32)
    P = regionprops(label(cellMask>0,connectivity=1),nucleiMask>0,cache=False)
    count=0
    for props in P:
         if props.max_intensity>0 :
            count += 1
            yi = props.coords[:, 0]
            xi = props.coords[:, 1]
            finalCellMask[yi, xi] = count
    nucleiMask = np.array(nucleiMask>0,dtype=np.uint32)
    nucleiMask = finalCellMask*nucleiMask
    cytoplasmMask = np.subtract(finalCellMask,nucleiMask)
    return cytoplasmMask,nucleiMask,finalCellMask
    
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
        imsave(outputPath + os.path.sep + fileName + 'Mask.tif',mask, plugin="tifffile")
        
    if saveFig== True:
        mask=np.uint8(mask>0)
        edges=cv2.Canny(mask,0,1)
        stacked_img=np.stack((np.uint16(edges)*65535,image),axis=0)
        tifffile.imsave(outputPath + os.path.sep + fileName + 'Outlines.tif',stacked_img)
        
    
        
    # assign nan to tissue mask


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
    nucleiProbMaps = tifffile.imread(nucleiClassProbPath,key=0)
    nucleiPM = nucleiProbMaps[int(PMrect[0]):int(PMrect[0]+PMrect[2]), int(PMrect[1]):int(PMrect[1]+PMrect[3])]
    nucleiProbMaps = tifffile.imread(contoursClassProbPath,key=0)
    PMSize = nucleiProbMaps.shape
    nucleiPM = np.dstack((nucleiPM,nucleiProbMaps[int(PMrect[0]):int(PMrect[0]+PMrect[2]), int(PMrect[1]):int(PMrect[1]+PMrect[3])]))

    # mask the core/tissue
    print(datetime.datetime.now(), 'Computing tissue mask')
    if args.crop == 'dearray':
        TMAmask = tifffile.imread(maskPath)
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
    nucleiMask = S3NucleiSegmentationWatershed(nucleiPM,nucleiCrop,args.logSigma,TMAmask,args.nucleiFilter,args.nucleiRegion)
    del nucleiPM
    # cytoplasm segmentation
    print(datetime.datetime.now(), 'Segmenting cytoplasm')
    if args.segmentCytoplasm == 'segmentCytoplasm':
        count =0
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
  
        cytoplasmMask,nucleiMaskTemp,cellMask = S3CytoplasmSegmentation(nucleiMask,cyto,TMAmask,'ring',args.cytoDilation)
        exportMasks(nucleiMaskTemp,nucleiCrop,outputPath,filePrefix,'nucleiRing',args.saveFig,args.saveMask)
        exportMasks(cytoplasmMask,cyto,outputPath,filePrefix,'cytoRing',args.saveFig,args.saveMask)
        exportMasks(cellMask,cyto,outputPath,filePrefix,'cellRing',args.saveFig,args.saveMask)
        
    elif args.segmentCytoplasm == 'ignoreCytoplasm':
        exportMasks(nucleiMask,nucleiCrop,outputPath,filePrefix,'nuclei')
        cellMask = nucleiMask
        exportMasks(nucleiMask,nucleiCrop,outputPath,filePrefix,'cell')
        cytoplasmMask = nucleiMask
        
        #fix bwdistance watershed
   
                