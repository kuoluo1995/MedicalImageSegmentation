import vtk


def main():
    colors = vtk.vtkNamedColors()

    colors.SetColor('leftBkg', [0.6, 0.5, 0.4, 1.0])
    colors.SetColor('centreBkg', [0.3, 0.1, 0.4, 1.0])
    colors.SetColor('rightBkg', [0.4, 0.5, 0.6, 1.0])

    sphereSource = vtk.vtkSphereSource()
    sphereSource.Update()

    print("There are %s input points" % sphereSource.GetOutput().GetNumberOfPoints())
    print("There are %s input cells" % sphereSource.GetOutput().GetNumberOfCells())

    ids = vtk.vtkIdTypeArray()
    ids.SetNumberOfComponents(1)

    # Specify that we want to extract cells 10 through 19
    i = 10
    while i < 20:
        ids.InsertNextValue(i)
        i += 1

    selectionNode = vtk.vtkSelectionNode()
    selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
    selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
    selectionNode.SetSelectionList(ids)

    selection = vtk.vtkSelection()
    selection.AddNode(selectionNode)

    extractSelection = vtk.vtkExtractSelection()
    extractSelection.SetInputConnection(0, sphereSource.GetOutputPort())
    extractSelection.SetInputData(1, selection)
    extractSelection.Update()

    # In selection
    selected = vtk.vtkUnstructuredGrid()
    selected.ShallowCopy(extractSelection.GetOutput())

    print("There are %s points in the selection" % selected.GetNumberOfPoints())
    print("There are %s cells in the selection" % selected.GetNumberOfCells())

    # Get points that are NOT in the selection
    selectionNode.GetProperties().Set(vtk.vtkSelectionNode.INVERSE(), 1)  # invert the selection
    extractSelection.Update()

    notSelected = vtk.vtkUnstructuredGrid()
    notSelected.ShallowCopy(extractSelection.GetOutput())

    print("There are %s points NOT in the selection" % notSelected.GetNumberOfPoints())
    print("There are %s cells NOT in the selection" % notSelected.GetNumberOfCells())

    backfaces = vtk.vtkProperty()
    backfaces.SetColor(colors.GetColor3d('Red'))

    inputMapper = vtk.vtkDataSetMapper()
    inputMapper.SetInputConnection(sphereSource.GetOutputPort())
    inputActor = vtk.vtkActor()
    inputActor.SetMapper(inputMapper)
    inputActor.SetBackfaceProperty(backfaces)

    selectedMapper = vtk.vtkDataSetMapper()
    selectedMapper.SetInputData(selected)

    selectedActor = vtk.vtkActor()
    selectedActor.SetMapper(selectedMapper)
    selectedActor.SetBackfaceProperty(backfaces)

    notSelectedMapper = vtk.vtkDataSetMapper()
    notSelectedMapper.SetInputData(notSelected)

    notSelectedActor = vtk.vtkActor()
    notSelectedActor.SetMapper(notSelectedMapper)
    notSelectedActor.SetBackfaceProperty(backfaces)

    # There will be one render window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(900, 300)

    # And one interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)

    # Define viewport ranges
    # (xmin, ymin, xmax, ymax)
    leftViewport = [0.0, 0.0, 0.5, 1.0]
    centerViewport = [0.33, 0.0, .66, 1.0]
    rightViewport = [0.66, 0.0, 1.0, 1.0]

    # Create a camera for all renderers
    camera = vtk.vtkCamera()

    # Setup the renderers
    leftRenderer = vtk.vtkRenderer()
    renderWindow.AddRenderer(leftRenderer)
    leftRenderer.SetViewport(leftViewport)
    leftRenderer.SetBackground(colors.GetColor3d('leftBkg'))
    leftRenderer.SetActiveCamera(camera)

    centerRenderer = vtk.vtkRenderer()
    renderWindow.AddRenderer(centerRenderer)
    centerRenderer.SetViewport(centerViewport)
    centerRenderer.SetBackground(colors.GetColor3d('centreBkg'))
    centerRenderer.SetActiveCamera(camera)

    rightRenderer = vtk.vtkRenderer()
    renderWindow.AddRenderer(rightRenderer)
    rightRenderer.SetViewport(rightViewport)
    rightRenderer.SetBackground(colors.GetColor3d('rightBkg'))
    rightRenderer.SetActiveCamera(camera)

    leftRenderer.AddActor(inputActor)
    centerRenderer.AddActor(selectedActor)
    rightRenderer.AddActor(notSelectedActor)

    leftRenderer.ResetCamera()

    renderWindow.Render()
    interactor.Start()


if __name__ == '__main__':
    main()