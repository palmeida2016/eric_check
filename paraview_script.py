# trace generated using paraview version 5.6.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# find source
artscase = FindSource('arts.case')

# create a new 'Slice'
slice1 = Slice(Input=artscase)
slice1.SliceType = 'Plane'
slice1.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Origin = [0.1999999969266355, -0.4123305678367615, -1.2000001668930054]

# Properties modified on slice1.SliceType
slice1.SliceType.Normal = [0.0, 1.0, 0.0]

# Properties modified on slice1.SliceType
slice1.SliceType.Normal = [0.0, 1.0, 0.0]

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1293, 547]

# show data in view
slice1Display = Show(slice1, renderView1)

# get color transfer function/color map for 'U'
uLUT = GetColorTransferFunction('U')

# trace defaults for the display properties.
slice1Display.Representation = 'Surface'
slice1Display.ColorArrayName = ['POINTS', 'U']
slice1Display.LookupTable = uLUT
slice1Display.OSPRayScaleArray = 'U'
slice1Display.OSPRayScaleFunction = 'PiecewiseFunction'
slice1Display.SelectOrientationVectors = 'Diss'
slice1Display.ScaleFactor = 0.039374999422580007
slice1Display.SelectScaleArray = 'U'
slice1Display.GlyphType = 'Arrow'
slice1Display.GlyphTableIndexArray = 'U'
slice1Display.GaussianRadius = 0.0019687499711290002
slice1Display.SetScaleArray = ['POINTS', 'U']
slice1Display.ScaleTransferFunction = 'PiecewiseFunction'
slice1Display.OpacityArray = ['POINTS', 'U']
slice1Display.OpacityTransferFunction = 'PiecewiseFunction'
slice1Display.DataAxesGrid = 'GridAxesRepresentation'
slice1Display.SelectionCellLabelFontFile = ''
slice1Display.SelectionPointLabelFontFile = ''
slice1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
slice1Display.DataAxesGrid.XTitleFontFile = ''
slice1Display.DataAxesGrid.YTitleFontFile = ''
slice1Display.DataAxesGrid.ZTitleFontFile = ''
slice1Display.DataAxesGrid.XLabelFontFile = ''
slice1Display.DataAxesGrid.YLabelFontFile = ''
slice1Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
slice1Display.PolarAxes.PolarAxisTitleFontFile = ''
slice1Display.PolarAxes.PolarAxisLabelFontFile = ''
slice1Display.PolarAxes.LastRadialAxisTextFontFile = ''
slice1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get opacity transfer function/opacity map for 'U'
uPWF = GetOpacityTransferFunction('U')

# set active source
SetActiveSource(artscase)

# create a new 'Slice'
slice2 = Slice(Input=artscase)
slice2.SliceType = 'Plane'
slice2.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice2.SliceType.Origin = [0.1999999969266355, -0.4123305678367615, -1.2000001668930054]

# Properties modified on slice2.SliceType
slice2.SliceType.Origin = [0.1999999969266355, -0.42, -1.2000001668930054]
slice2.SliceType.Normal = [0.0, 1.0, 0.0]

# Properties modified on slice2.SliceType
slice2.SliceType.Origin = [0.1999999969266355, -0.42, -1.2000001668930054]
slice2.SliceType.Normal = [0.0, 1.0, 0.0]

# show data in view
slice2Display = Show(slice2, renderView1)

# trace defaults for the display properties.
slice2Display.Representation = 'Surface'
slice2Display.ColorArrayName = ['POINTS', 'U']
slice2Display.LookupTable = uLUT
slice2Display.OSPRayScaleArray = 'U'
slice2Display.OSPRayScaleFunction = 'PiecewiseFunction'
slice2Display.SelectOrientationVectors = 'Diss'
slice2Display.ScaleFactor = 0.039374999422580007
slice2Display.SelectScaleArray = 'U'
slice2Display.GlyphType = 'Arrow'
slice2Display.GlyphTableIndexArray = 'U'
slice2Display.GaussianRadius = 0.0019687499711290002
slice2Display.SetScaleArray = ['POINTS', 'U']
slice2Display.ScaleTransferFunction = 'PiecewiseFunction'
slice2Display.OpacityArray = ['POINTS', 'U']
slice2Display.OpacityTransferFunction = 'PiecewiseFunction'
slice2Display.DataAxesGrid = 'GridAxesRepresentation'
slice2Display.SelectionCellLabelFontFile = ''
slice2Display.SelectionPointLabelFontFile = ''
slice2Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
slice2Display.DataAxesGrid.XTitleFontFile = ''
slice2Display.DataAxesGrid.YTitleFontFile = ''
slice2Display.DataAxesGrid.ZTitleFontFile = ''
slice2Display.DataAxesGrid.XLabelFontFile = ''
slice2Display.DataAxesGrid.YLabelFontFile = ''
slice2Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
slice2Display.PolarAxes.PolarAxisTitleFontFile = ''
slice2Display.PolarAxes.PolarAxisLabelFontFile = ''
slice2Display.PolarAxes.LastRadialAxisTextFontFile = ''
slice2Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# show color bar/color legend
slice2Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# hide data in view
Hide(artscase, renderView1)

# set active source
SetActiveSource(artscase)

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.CameraPosition = [0.3790551231361666, -0.06760006377388889, -0.533826417839061]
renderView1.CameraFocalPoint = [0.1999999969266355, -0.41233056783676153, -1.2000001668930058]
renderView1.CameraViewUp = [-0.05526185096622922, 0.8927599605390729, -0.44713060808454985]
renderView1.CameraParallelScale = 0.2922209251752841

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).