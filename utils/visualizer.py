import camviz as cv

class RefinementVisualizer:
    def __init__(self, env):
        self.env = env
        self.draw = cv.Draw(wh=(1600, 900))
        self._setup_panels()

    def _setup_panels(self):
        self.draw.add2Dimage('rgb', (0.0, 0.0, 0.33, 0.5))
        self.draw.add2Dimage('depth', (0.0, 0.5, 0.33, 1.0))
        self.draw.add3Dworld('3d_view', (0.33, 0.0, 1.0, 1.0))

    def update(self, refined_depth):
        self.draw['rgb'].image(self.env.rgb)
        self.draw['depth'].image(cv.utils.viz_depth(refined_depth))
        self._update_3d_view(refined_depth)

    def _update_3d_view(self, depth):
        points = self.env._depth_to_3d(depth)
        self.draw['3d_view'].points(points)
