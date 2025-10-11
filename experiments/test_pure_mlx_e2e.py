#!/usr/bin/env python3
"""
Step 5: End-to-end test of Pure MLX implementation
Goal: Verify that pure MLX mode can generate audio (even if quality is not perfect yet)
"""

import sys
sys.path.insert(0, '/Users/bailin/index-tts')

import warnings
warnings.filterwarnings('ignore')

import time
from indextts.infer_v2 import IndexTTS2

print("\n" + "="*70)
print("Step 5: Pure MLX End-to-End Test")
print("="*70)
print("\nGoal: Verify pure MLX pipeline can generate audio")
print("Note: Quality may not be perfect yet (conditioning weights not loaded)\n")

def test_pure_mlx_generation():
    """Test pure MLX generation pipeline"""
    
    print("="*70)
    print("Test: Pure MLX Audio Generation")
    print("="*70)
    
    # Initialize with MLX
    print("\n1. Initializing IndexTTS2 with Pure MLX...")
    try:
        start = time.time()
        tts = IndexTTS2(
            ckpt_dir="/Users/bailin/index-tts/checkpoints",
            use_mlx=True  # Enable MLX
        )
        init_time = time.time() - start
        print(f"✓ Initialization complete ({init_time:.2f}s)")
    except Exception as e:
        print(f"❌ FAIL: Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test cases
    test_cases = [
        ("今天", "Short text"),
        ("你好", "Short greeting"),
    ]
    
    results = []
    
    for text, desc in test_cases:
        print(f"\n{'='*70}")
        print(f"2. Generating: \"{text}\" ({desc})")
        print(f"{'='*70}")
        
        try:
            start = time.time()
            audio_data, sample_rate = tts(
                text=text,
                ref_audio="/Users/bailin/index-tts/examples/参考音频.wav",
                ref_text="对,我们都是有梦想的。",
                emo_audio="/Users/bailin/index-tts/examples/参考音频.wav",
                emo_text="对,我们都是有梦想的。",
            )
            gen_time = time.time() - start
            
            # Check results
            if audio_data is None:
                print(f"❌ FAIL: Generated audio is None")
                results.append((text, False, 0, 0))
                continue
            
            duration = len(audio_data) / sample_rate
            audio_std = audio_data.std()
            audio_mean = abs(audio_data.mean())
            audio_max = abs(audio_data).max()
            
            print(f"\n✓ Generation successful:")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Time: {gen_time:.2f}s (RTF={gen_time/duration:.2f}x)")
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Audio shape: {audio_data.shape}")
            print(f"  Audio std: {audio_std:.4f}")
            print(f"  Audio mean: {audio_mean:.4f}")
            print(f"  Audio max: {audio_max:.4f}")
            
            # Quality checks
            issues = []
            
            # Check 1: Not silent
            if audio_std < 0.001:
                issues.append("Audio is nearly silent (std < 0.001)")
            
            # Check 2: Not clipping
            if audio_max > 0.99:
                issues.append("Audio may be clipping (max > 0.99)")
            
            # Check 3: Duration reasonable (1-10s for short text)
            if duration < 0.5 or duration > 15.0:
                issues.append(f"Duration unusual: {duration:.2f}s")
            
            # Check 4: Has variation
            if audio_data.max() - audio_data.min() < 0.01:
                issues.append("Audio has very low dynamic range")
            
            if issues:
                print(f"\n⚠️  Quality Issues:")
                for issue in issues:
                    print(f"    - {issue}")
                results.append((text, "warning", duration, gen_time))
            else:
                print(f"\n✅ Quality checks passed")
                results.append((text, True, duration, gen_time))
            
            # Save audio
            output_path = f"experiments/pure_mlx_{text}.wav"
            import scipy.io.wavfile as wavfile
            wavfile.write(output_path, sample_rate, (audio_data * 32767).astype('int16'))
            print(f"  Saved: {output_path}")
            
        except Exception as e:
            print(f"\n❌ FAIL: Generation failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((text, False, 0, 0))
    
    return results


def test_pure_mlx_vs_hybrid():
    """Compare Pure MLX vs Hybrid mode"""
    print("\n" + "="*70)
    print("Test: Pure MLX vs Hybrid Comparison")
    print("="*70)
    
    # This test requires switching conditioning mode, which would need code changes
    # For now, just verify pure MLX works
    print("\n⚠️  Skipping comparison test (requires hybrid mode)")
    print("    Pure MLX conditioning uses randomly initialized weights")
    print("    Quality will improve after loading PyTorch weights in Step 6")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Pure MLX End-to-End Validation")
    print("="*70)
    
    # Test 1: Generation
    try:
        results = test_pure_mlx_generation()
        
        if not results:
            print("\n❌ Generation test failed completely")
            sys.exit(1)
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        successful = sum(1 for r in results if r[1] == True)
        warnings = sum(1 for r in results if r[1] == "warning")
        failed = sum(1 for r in results if r[1] == False)
        
        print(f"\nResults: {successful} passed, {warnings} warnings, {failed} failed")
        
        for text, status, duration, gen_time in results:
            if status == True:
                print(f"  ✅ \"{text}\": {duration:.2f}s audio in {gen_time:.2f}s")
            elif status == "warning":
                print(f"  ⚠️  \"{text}\": {duration:.2f}s audio in {gen_time:.2f}s (quality issues)")
            else:
                print(f"  ❌ \"{text}\": Failed")
        
        if successful > 0 or warnings > 0:
            print("\n" + "="*70)
            print("🎉 Pure MLX pipeline is functional!")
            print("="*70)
            print("\n📝 Next Steps:")
            print("  1. Load Conformer + Perceiver weights from PyTorch")
            print("  2. Verify conditioning quality matches PyTorch")
            print("  3. Performance optimization")
            print("\n" + "="*70)
        else:
            print("\n❌ Pure MLX pipeline has critical issues")
        
    except Exception as e:
        print(f"\n❌ Test crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

