import vtk

source = vtk.vtkSphereSource()
source.SetCenter(0, 0, 0)
source.SetRadius(1)
source.Update()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(source.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer = vtk.vtkRenderer()
renderer.SetBackground(1, 1, 1)
renderer.AddActor(actor)

renwin = vtk.vtkRenderWindow()
renwin.AddRenderer(renderer)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
interactor.SetRenderWindow(renwin)

def DummyFunc1(obj, ev):
    print("Before Event")

def DummyFunc2(obj, ev):
    print("After Event")

# Print interator gives you a list of registered observers of the current
# interactor style
#print(interactor)


## adding priorities allow to control the order of observer execution
## (highest value first! if equal the first added observer is called first)
interactor.RemoveObservers('RightButtonPressEvent')
interactor.AddObserver('RightButtonPressEvent', DummyFunc1, 1.0)
interactor.AddObserver('RightButtonPressEvent', DummyFunc2, -1.0)
interactor.Initialize()
interactor.Start()