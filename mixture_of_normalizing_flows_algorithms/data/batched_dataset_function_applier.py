class BatchedDatasetFunctionApplier:
    def __init__(self, batched_dataset, function):
        self.batched_dataset = batched_dataset
        self.function = function

    def apply_return_generator(self):
        for batch in self.batched_dataset:
            yield self.function(batch)
