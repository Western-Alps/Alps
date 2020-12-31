#ifndef ITKHEADERS_H
#define ITKHEADERS_H
//
// ITK
//
#include <itkBSplineInterpolateImageFunction.h>
#include <itkChangeInformationImageFilter.h>
#include <itkConstNeighborhoodIterator.h>
#include <itkIdentityTransform.h>
#include <itkImageDuplicator.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImage.h>
#include <itkImageRegionIterator.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkNiftiImageIO.h>
#include <itkOrientImageFilter.h>
#include <itkReLUInterpolateImageFunction.h>
#include <itkResampleImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkShrinkImageFilter.h>
#include <itkSize.h>
#include <itkSpatialOrientation.h>
// Some typedef
template< int Dim > using ImageType = itk::Image< double, Dim >;
template< int Dim > using Reader    = itk::ImageFileReader< ImageType< Dim > >;
template< int Dim > using Writer    = itk::ImageFileWriter< ImageType< Dim > >;
template< int Dim > using MaskType  = itk::Image< unsigned char, Dim >;
//
template< int Dim > using ConvolutionWindowType = itk::Size< Dim >;
template< int Dim > using DuplicatorType        = itk::ImageDuplicator< ImageType< Dim > > ;
template< int Dim > using FilterType            = itk::ChangeInformationImageFilter< ImageType< Dim > >;
template< int Dim > using ShrinkImageFilterType = itk::ShrinkImageFilter < ImageType< Dim >, ImageType< Dim > >;
template< int Dim > using ShrinkImageFilterType = itk::ShrinkImageFilter < ImageType< Dim >, ImageType< Dim > >;
//
template< typename ImgType > using ImageRegionIterator =  itk::ImageRegionIterator< ImgType >;
//
#endif
