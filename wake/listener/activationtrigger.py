class ActivationTrigger:
    activation_ctr: int
    sensitivity: float
    trigger_level: int
    activation_delay: int

    def __init__(self, sensitivity=0.3, trigger_level=3, activation_delay=8):
        self.sensitivity = sensitivity
        self.trigger_level = trigger_level
        self.activation_delay = activation_delay
        self.activation_ctr = 0

    def check_trigger(self, prob) -> bool:
        # Check if the probability is above the sensitivity threshold
        chunk_activated = prob > 1.0 - self.sensitivity

        # If the chunk is activated, increment the activation counter
        if chunk_activated or self.activation_ctr < 0:
            self.activation_ctr += 1

            # If the activation counter is above the trigger level, return True
            has_activated = self.activation_ctr > self.trigger_level
            
            # If the chunk is activated, reset the activation counter
            if has_activated or chunk_activated and self.activation_ctr < 0:
                # Wait for activation_delay chunks after trigger without any 
                # activations before counting new activations
                self.activation_ctr = -self.activation_delay

            if has_activated:
                return True
            
        elif self.activation_ctr > 0:
            self.activation_ctr -= 1
        return False

