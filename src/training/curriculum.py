class CurriculumScheduler:
    """
    Manages the Curriculum Learning phases: 
    Dry Synth -> Non-Linear FX -> Full Chain.
    """
    def __init__(self, phases=None):
        self.phases = phases or [
            {"name": "Dry Synth", "epochs": 10, "complexity": 0.0},     # Only optimize Synth params
            {"name": "Non-Linear FX", "epochs": 20, "complexity": 0.5}, # Add some FX
            {"name": "Full Chain", "epochs": 100, "complexity": 1.0}    # Full End-to-End
        ]
        self.current_phase_idx = 0
        self.current_epoch = 0

    def step(self):
        """
        updates the internal state at the end of an epoch.
        """
        self.current_epoch += 1
        
        # Check if we should advance phase
        # Calculate cumulative epochs for boundaries
        cumulative_epochs = 0
        for i, phase in enumerate(self.phases):
            cumulative_epochs += phase["epochs"]
            if self.current_epoch < cumulative_epochs:
                self.current_phase_idx = i
                return
        
        # If we exceed all phases, stay in the last one
        self.current_phase_idx = len(self.phases) - 1

    def get_current_phase(self):
        return self.phases[self.current_phase_idx]

    def get_loss_weights(self):
        """
        Returns dynamic weights for losses based on phase.
        e.g., in Dry phase, Spectral Loss might be weighted differently or 
        we might mask out FX parameters in the loss.
        """
        phase = self.get_current_phase()
        
        # Example logic:
        # Phase 0: Focus purely on matching specific parameters or simple spectral features?
        # Actually, for "Dry Synth", we probably want to disable the FX part of the Proxy 
        # or mask gradients for FX params.
        
        return {
            "name": phase["name"],
            "complexity_factor": phase["complexity"]
        }
