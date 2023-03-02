import numpy as np

class EarlyStopper:
    """
    Stops the training early iff the validation loss does not increase anymore more than a small delta.
    As the FLOE is weighted only half as much as the other two, the model does only need to get better half as much there.
    """
    def __init__(self, patience=1, min_delta=0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_val_score = -np.inf
        
    def early_stop(self, combined_score, best_combined_score):
        # does not improve enough anymore: < old best + delta
        if combined_score <= (self.max_val_score + self.min_delta):
            self.counter += 1
        else:
            self.max_val_score = best_combined_score # such that there is not a crawling improvement not seen
            self.counter = 0
        return self.counter > self.patience