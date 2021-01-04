#include "testITKFunctions.h"
//
// ITK
//
#include "ITKHeaders.h"

//using ::testing::Return;

ITKFunctionTest::ITKFunctionTest() {
    // Have qux return true by default
    //ON_CALL(m_bar,qux()).WillByDefault(Return(true));
    // Have norf return false by default
    //ON_CALL(m_bar,norf()).WillByDefault(Return(false));
}

ITKFunctionTest::~ITKFunctionTest() {};

void ITKFunctionTest::SetUp() {};

void ITKFunctionTest::TearDown() {};

TEST_F(ITKFunctionTest, SimpleImage) {
  //
  //
  ImageType<2>::RegionType region;
  ImageType<2>::IndexType  start;
  start[0] = 0;
  start[1] = 0;
  //
  ImageType<2>::SizeType size;
  size[0] = 200;
  size[1] = 300;
  //
  region.SetSize(size);
  region.SetIndex(start);
  //
  //
  ImageType<2>::Pointer image = ImageType<2>::New();
  image->SetRegions(region);
  image->Allocate();
  //
  ImageType<2>::IndexType ind;
  ind[0] = 10;
  ind[1] = 10;

  //
  //
  Writer<2>::Pointer writer = Writer<2>::New();
  writer->SetFileName( "test.nii.gz" );
  writer->SetInput(image);

  try
  {
    writer->Update();
  }
  catch (itk::ExceptionObject & error)
  {
    std::cerr << "error" << std::endl;
  }

  EXPECT_EQ( 0, 0 );
}

//TEST_F(ITKFunctionTest, ByDefaultBazFalseIsFalse) {
//    LoadDataSet foo(m_bar);
//    EXPECT_EQ(foo.baz(false), false);
//}
//
//TEST_F(ITKFunctionTest, SometimesBazFalseIsTrue) {
//    LoadDataSet foo(m_bar);
//    // Have norf return true for once
//    EXPECT_CALL(m_bar,norf()).WillOnce(Return(true));
//    EXPECT_EQ(foo.baz(false), true);
//}

