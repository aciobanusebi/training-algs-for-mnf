class BaseDummyDistributionCreator:
    def __init__(self, distribution):
        self.distribution = distribution

    def create(self):
        return self.distribution
