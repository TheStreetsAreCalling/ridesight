from panda3d.core import loadPrcFileData
from panda3d.core import DirectionalLight, AmbientLight
from panda3d.core import Vec3, Vec4, WindowProperties
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
import time

# ---- Monterey OpenGL compatibility ----
loadPrcFileData('', 'load-display pandagl')
loadPrcFileData('', 'gl-version 2 1')
loadPrcFileData('', 'glsl-version 120')
loadPrcFileData('', 'basic-shaders-only true')
loadPrcFileData('', 'framebuffer-srgb false')


class MyApp(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)
        self.disableMouse()

        # -----------------------
        # Mouse / camera settings
        # -----------------------
        self.sensitivity = 0.2
        self.cam_heading = 132.61     # yaw
        self.cam_pitch = 50 # pitch
        self.cam_roll = 0 # roll
        self.cam_distance = 10   # distance behind car

        # Print throttle
        self.last_print_time = 0.0
        self.print_interval = 0.1

        # Lock mouse
        props = WindowProperties()
        props.setCursorHidden(True)
        self.win.requestProperties(props)

        self.center_x = self.win.getXSize() // 2
        self.center_y = self.win.getYSize() // 2
        self.win.movePointer(0, self.center_x, self.center_y)

        # -----------------------
        # Lighting
        # -----------------------
        dlight = DirectionalLight("dlight")
        dlight.setColor(Vec4(1, 1, 1, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(45, -45, 0)
        self.render.setLight(dlnp)

        alight = AmbientLight("alight")
        alight.setColor(Vec4(0.7, 0.7, 0.7, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)

        # -----------------------
        # Car (GLB)
        # -----------------------
        self.car = self.loader.loadModel("car.glb")
        self.car.reparentTo(self.render)
        self.car.setScale(0.45)
        self.car.setPos(0, 0, 0)
        self.car.setHpr(0, 0, 0)

        # -----------------------
        # Camera pivot (GTA style)
        # -----------------------
        self.cam_pivot = self.car.attachNewNode("cam_pivot")
        self.cam_pivot.setPos(0, 0, 1.5)  # height above car

        self.camera.reparentTo(self.cam_pivot)
        self.camera.setPos(15, -self.cam_distance - 6, 20)
        self.camera.lookAt(self.car)

        # Task
        self.taskMgr.add(self.camera_task, "CameraTask")

    def camera_task(self, task):
        if self.mouseWatcherNode.hasMouse():
            md = self.win.getPointer(0)
            dx = md.getX() - self.center_x
            dy = md.getY() - self.center_y

            # Update angles
            self.cam_heading -= dx * self.sensitivity
            self.cam_pitch -= dy * self.sensitivity
            self.cam_pitch = max(-60, min(10, self.cam_pitch))

            # Apply rotation to pivot
            self.cam_pivot.setHpr(self.cam_heading, self.cam_pitch, self.cam_roll)

            # Print camera rotation
            now = time.time()
            if now - self.last_print_time >= self.print_interval:
                h, p, r = self.cam_pivot.getHpr()
                print(f"Camera orbit -> H: {h:.2f}°, P: {p:.2f}°, R: {r:.2f}°")
                self.last_print_time = now

            # Re-center mouse
            self.win.movePointer(0, self.center_x, self.center_y)

        return Task.cont


app = MyApp()
app.run()

