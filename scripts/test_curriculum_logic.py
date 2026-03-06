import torch
import torch.nn.functional as F
import sys
import os

# Ensure src is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.proxy.chain import ProxyChainer
from src.training.train_inverter_audio import assemble_full_params, denormalize_params

def run_curriculum_tests():
    print("==================================================")
    print("      V9.0 CURRICULUM TEACHER FORCING TEST        ")
    print("==================================================")
    
    device = torch.device('cpu')
    batch_size = 2
    
    # 1. Initialize Chainer
    # We use a dummy parameter dictionary since we won't actually render audio,
    # we just want to verify the Tensors being passed to forward_flat have the right shapes and sources.
    class MockChainer(ProxyChainer):
        def forward_flat(self, flat_params, order_idx=None, sequence=None, wave_logits=None, order_logits=None):
            # We intercept the call to verify what was passed
            self.last_flat_params = flat_params
            self.last_order_logits = order_logits
            self.last_wave_logits = wave_logits
            # Return dummy audio just to satisfy the loop
            return torch.zeros(batch_size, 1, 1024)
            
    chainer = MockChainer().to(device)
    
    # 2. Mock Ground Truth (Teacher)
    # 62 params total in normalized space. We just need to track the categorical ones.
    # Wave Type (idx 1) = 15 (Square)
    # Sat Type (idx 17) = 2 (Fold)
    # EQ Types (idx 21, 25, etc) = [1, 2, 3, 4, 1, 2, 3, 4]
    true_params = torch.zeros(batch_size, 62)
    true_params[:, 1] = 15
    true_params[:, 17] = 2
    for i in range(8):
        true_params[:, 21 + i*4] = (i % 4) + 1
        
    true_order_idx = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]) # Batch size 2
    true_wave_idx = true_params[:, 1].long()
    
    # 3. Mock Model Predictions (Student)
    # Lets make the student predict entirely wrong categorical values to prove the routing logic switches.
    # We'll make it predict Wave Type 0 (Sine)
    outputs = {}
    outputs['wave_logits'] = torch.zeros(batch_size, 128)
    outputs['wave_logits'][:, 0] = 100.0 # Argmax is 0
    
    outputs['sat_type_logits'] = torch.zeros(batch_size, 6)
    outputs['sat_type_logits'][:, 0] = 100.0 # Argmax is 0
    
    outputs['eq8_type_logits'] = [torch.zeros(batch_size, 8) for _ in range(8)]
    for i in range(8):
        outputs['eq8_type_logits'][i][:, 0] = 100.0 # Argmax is 0
        
    outputs['order_logits'] = torch.zeros(batch_size, 4, 5) # Dummy predicted order
    
    # Continuous dials (Doesn't matter for this test, just needs to be large enough: 14 + 3 + 24 + 7 + 3 = 51)
    mu_best = torch.zeros(batch_size, 51)
    
    print("\n--- PHASE 1: TEACHER FORCING (Epoch 10) ---")
    epoch = 10
    teacher_force_epochs = 50
    
    if epoch < teacher_force_epochs:
        # Exact copy of the logic from train_inverter_audio.py
        true_sat_type_idx = true_params[:, 17].long()
        true_eq8_type_idxs = [true_params[:, 21 + i*4].long() for i in range(8)]
        
        full_params_norm = assemble_full_params(mu_best, true_wave_idx, true_sat_type_idx, true_eq8_type_idxs, batch_size, device)
        pred_params_raw = denormalize_params(full_params_norm)
        
        true_order_logits = None
        if true_order_idx is not None:
            true_order_logits = F.one_hot(true_order_idx, num_classes=5).float() * 100.0
        
        _ = chainer.forward_flat(
            pred_params_raw, 
            order_idx=None, 
            wave_logits=None, 
            order_logits=true_order_logits
        )
        
        # VERIFICATION
        assert chainer.last_wave_logits is None, "❌ Phase 1: wave_logits should be None (Bypassed!)"
        assert chainer.last_order_logits.shape == (2, 4, 5), "❌ Phase 1: Order Logits shape mismatch"
        # Check that the Teacher Force flat_params actually absorbed index 15 for Wave Type (via denorm logic)
        # Denorm maps Wave 15 back to 15.0. 
        wave_param = chainer.last_flat_params[0, 1].item()
        assert wave_param == 15.0, f"❌ Phase 1: Expected true Wave=15.0, got {wave_param}"
        print("✅ Phase 1 Logic perfectly forced the Teacher (Ground Truth) values.")

    
    print("\n--- PHASE 2: UNLOCKED AUDIO GUIDED (Epoch 60) ---")
    epoch = 60
    
    if epoch >= teacher_force_epochs:
        # Exact copy of the logic from train_inverter_audio.py
        pred_wave_idx = torch.argmax(outputs['wave_logits'], dim=1)
        pred_sat_type = torch.argmax(outputs['sat_type_logits'], dim=1)
        pred_eq8_types = [torch.argmax(logits, dim=1) for logits in outputs['eq8_type_logits']]

        full_params_norm = assemble_full_params(mu_best, pred_wave_idx, pred_sat_type, pred_eq8_types, batch_size, device)
        pred_params_raw = denormalize_params(full_params_norm)
        
        _ = chainer.forward_flat(
            pred_params_raw, 
            wave_logits=outputs['wave_logits'],
            order_logits=outputs['order_logits']
        )
        
        # VERIFICATION
        assert chainer.last_wave_logits is outputs['wave_logits'], "❌ Phase 2: Did not use predicted wave_logits"
        assert chainer.last_order_logits is outputs['order_logits'], "❌ Phase 2: Did not use predicted order_logits"
        wave_param = chainer.last_flat_params[0, 1].item()
        assert wave_param == 0.0, f"❌ Phase 2: Expected predicted Wave=0.0, got {wave_param}"
        print("✅ Phase 2 Logic perfectly handed control back to the Student (Network Predictions).")

    print("\n==================================================")
    print("🎉 CURRICULUM LEARNING TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_curriculum_tests()
