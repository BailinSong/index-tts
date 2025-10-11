#!/usr/bin/env python3
"""
Simple standalone test for Step 4 & 5: Pure MLX Conditioning
Avoids import issues by testing components directly
"""

import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import mlx.core as mx
import mlx.nn as nn
import numpy as np

print("\n" + "="*70)
print("Step 4 & 5: Pure MLX Conditioning Integration Test")
print("="*70)

def step4_test_conditioning_creation():
    """Step 4: Test MLX conditioning module can be created"""
    print("\n" + "="*70)
    print("Step 4.1: Create Pure MLX Conditioning Module")
    print("="*70)
    
    try:
        from indextts.gpt.mlx_conditioning import MLXConditioningModule
        
        conditioning = MLXConditioningModule(
            input_dim=1024,
            model_dim=1280,
            num_latents=32,
            conformer_layers=4,
            perceiver_depth=2
        )
        
        print("✅ Successfully created MLX conditioning module")
        print(f"   - Conformer: {len(conditioning.conformer.blocks)} layers")
        print(f"   - Perceiver: {len(conditioning.perceiver.layers)} layers")
        return True, conditioning
        
    except Exception as e:
        print(f"❌ Failed to create conditioning module: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def step4_test_conditioning_forward():
    """Step 4.2: Test forward pass"""
    print("\n" + "="*70)
    print("Step 4.2: Test Conditioning Forward Pass")
    print("="*70)
    
    success, conditioning = step4_test_conditioning_creation()
    if not success:
        return False
    
    try:
        # Create input (speaker embedding)
        batch, seq, dim = 2, 121, 1024
        x = mx.random.normal((batch, seq, dim))
        lengths = mx.array([121, 100])
        
        print(f"Input: {x.shape}")
        
        # Forward pass
        latents = conditioning(x, lengths)
        
        print(f"Output latents: {latents.shape}")
        print(f"Expected: (2, 32, 1280)")
        
        # Checks
        if latents.shape != (2, 32, 1280):
            print(f"❌ Shape mismatch")
            return False
        
        if mx.any(mx.isnan(latents)) or mx.any(mx.isinf(latents)):
            print(f"❌ Contains NaN or Inf")
            return False
        
        std = float(latents.std())
        if std < 0.01 or std > 10.0:
            print(f"❌ Unusual std: {std}")
            return False
        
        print(f"✅ Forward pass successful")
        print(f"   - Latents std: {std:.4f}")
        print(f"   - Latents range: [{float(latents.min()):.4f}, {float(latents.max()):.4f}]")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def step4_test_mlx_gpt_creation():
    """Step 4.3: Test MLX GPT with pure conditioning"""
    print("\n" + "="*70)
    print("Step 4.3: Create UnifiedVoiceMLX with Pure Conditioning")
    print("="*70)
    
    try:
        from indextts.gpt.mlx_model import UnifiedVoiceMLX
        
        # Create model with pure MLX conditioning
        model = UnifiedVoiceMLX(
            layers=4,  # Smaller for testing
            model_dim=1280,
            heads=16,
            max_text_tokens=402,
            max_mel_tokens=2002,
            max_conditioning_inputs=3,
            mel_length_compression=1024,
            number_text_tokens=256,
            start_text_token=255,
            stop_text_token=0,
            number_mel_codes=8194,
            start_mel_token=8192,
            stop_mel_token=8193,
            use_mlx_conditioning=True  # Pure MLX!
        )
        
        print(f"✅ Successfully created UnifiedVoiceMLX")
        print(f"   - Transformer layers: {model.layers}")
        print(f"   - MLX conditioning: {model.use_mlx_conditioning}")
        print(f"   - Conditioning module: {type(model.conditioning_module).__name__}")
        
        return True, model
        
    except Exception as e:
        print(f"❌ Failed to create UnifiedVoiceMLX: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def step5_test_conditioning_inference():
    """Step 5: Test conditioning inference"""
    print("\n" + "="*70)
    print("Step 5.1: Test MLX Conditioning Inference")
    print("="*70)
    
    success, model = step4_test_mlx_gpt_creation()
    if not success:
        return False
    
    try:
        # Create speaker embedding
        batch = 1
        spk_seq = 121
        spk_emb = mx.random.normal((batch, 1024, spk_seq))  # (batch, dim, seq)
        cond_lengths = mx.array([spk_seq])
        
        print(f"Speaker embedding: {spk_emb.shape}")
        
        # Get conditioning using MLX
        latents = model.get_conditioning_mlx(spk_emb, cond_lengths)
        
        print(f"Conditioning latents: {latents.shape}")
        print(f"Expected: (1, 32, 1280)")
        
        # Checks
        if latents.shape != (1, 32, 1280):
            print(f"❌ Shape mismatch")
            return False
        
        std = float(latents.std())
        print(f"✅ Conditioning inference successful")
        print(f"   - Latents std: {std:.4f}")
        print(f"   - Latents range: [{float(latents.min()):.4f}, {float(latents.max()):.4f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Conditioning inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def step5_test_text_generation():
    """Step 5.2: Test text token generation with MLX conditioning"""
    print("\n" + "="*70)
    print("Step 5.2: Test Text Generation with Pure MLX Conditioning")
    print("="*70)
    
    success, model = step4_test_mlx_gpt_creation()
    if not success:
        return False
    
    try:
        # Create inputs
        batch = 1
        text_tokens = mx.array([[10, 20, 30, 40, 50]])  # Short text
        
        # Create conditioning
        spk_emb = mx.random.normal((batch, 1024, 100))
        cond_lengths = mx.array([100])
        
        print(f"Text tokens: {text_tokens.shape}")
        print(f"Speaker embedding: {spk_emb.shape}")
        
        # Get conditioning
        conditioning = model.get_conditioning_mlx(spk_emb, cond_lengths)
        print(f"Conditioning: {conditioning.shape}")
        
        # Generate (short)
        print("\nGenerating tokens...")
        output = model.simple_forward(
            text_tokens,
            conditioning=conditioning,
            max_length=20,  # Very short for testing
            temperature=0.8
        )
        
        print(f"\nGenerated tokens: {output.shape}")
        print(f"Token values: {np.array(output[0])}")
        
        # Check if it stops properly
        output_list = list(np.array(output[0]))
        if model.stop_mel_token in output_list:
            stop_idx = output_list.index(model.stop_mel_token)
            print(f"✅ Stop token found at position {stop_idx}")
        else:
            print(f"⚠️  Stop token not found (may hit max_length)")
        
        # Check token diversity
        unique_tokens = len(set(output_list))
        print(f"Unique tokens: {unique_tokens}/{len(output_list)}")
        
        if unique_tokens < 3:
            print(f"⚠️  Low token diversity (may indicate quality issues)")
        else:
            print(f"✅ Reasonable token diversity")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    results = []
    
    # Step 4 tests
    print("\n🚀 Starting Step 4: Integration Tests")
    results.append(("4.1: Module Creation", step4_test_conditioning_creation()[0]))
    results.append(("4.2: Forward Pass", step4_test_conditioning_forward()))
    results.append(("4.3: GPT Creation", step4_test_mlx_gpt_creation()[0]))
    
    # Step 5 tests  
    print("\n🚀 Starting Step 5: End-to-End Tests")
    results.append(("5.1: Conditioning Inference", step5_test_conditioning_inference()))
    results.append(("5.2: Text Generation", step5_test_text_generation()))
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    passed_count = sum(1 for r in results if r[1])
    total_count = len(results)
    
    print("\n" + "="*70)
    print(f"Results: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed!")
        print("✓ Step 4: Pure MLX Conditioning integrated")
        print("✓ Step 5: End-to-end pipeline functional")
        print("\n📝 Notes:")
        print("  - Conditioning uses randomly initialized weights")
        print("  - Generation quality will improve after loading PyTorch weights")
        print("  - Next: Step 6 - Load weights and optimize")
    elif passed_count > 0:
        print("\n⚠️  Some tests passed, but not all")
    else:
        print("\n❌ All tests failed")
    
    print("="*70 + "\n")
    
    sys.exit(0 if passed_count == total_count else 1)

