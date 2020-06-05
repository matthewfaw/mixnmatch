class SimpleTorchStepSchedule:
    def __init__(self, lr):
        self.lr = lr

    def get_step_size(self, t):
        return self.lr

