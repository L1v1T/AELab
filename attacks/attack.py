


class Attack(object):
    def __init__(self, **kwargs):
        pass

    def generate(self, model, x, **kwargs):
        error = "Sub-classes must implement 'generate' method."
        raise NotImplementedError(error)