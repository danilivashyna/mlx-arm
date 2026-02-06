// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

package com.mlxarm.demo

import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.mlxarm.MLXContext
import com.mlxarm.MLXDevice
import com.mlxarm.MLXOps

class MainActivity : AppCompatActivity() {
    private val TAG = "MLX-Demo"
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        val outputText = findViewById<TextView>(R.id.output_text)
        val resultBuilder = StringBuilder()
        
        resultBuilder.appendLine("=== MLX-ARM Demo ===\n")
        
        try {
            // List available devices
            resultBuilder.appendLine("📱 Available Devices:")
            val devices = MLXDevice.getAvailableDevices()
            for (device in devices) {
                resultBuilder.appendLine("  • $device")
            }
            resultBuilder.appendLine()
            
            // Initialize MLX
            val mlx = MLXContext()
            val initialized = mlx.initialize()
            
            if (initialized) {
                resultBuilder.appendLine("✅ MLX initialized successfully")
                resultBuilder.appendLine("Device: ${mlx.getDeviceName()}")
                resultBuilder.appendLine("FP16 support: ${mlx.supportsFP16()}")
                resultBuilder.appendLine()
                
                // Run vector addition demo
                resultBuilder.appendLine("🧮 Running Vector Addition Demo...")
                val N = 1024
                val a = FloatArray(N) { it.toFloat() }
                val b = FloatArray(N) { (it * 2).toFloat() }
                
                val startTime = System.nanoTime()
                val c = MLXOps.nativeVectorAdd(mlx.hashCode().toLong(), a, b)
                val endTime = System.nanoTime()
                
                val duration = (endTime - startTime) / 1_000_000.0 // ms
                
                // Verify results
                var correct = true
                for (i in 0 until N) {
                    if (Math.abs(c[i] - (a[i] + b[i])) > 1e-5) {
                        correct = false
                        break
                    }
                }
                
                if (correct) {
                    resultBuilder.appendLine("✅ Results verified correctly!")
                    resultBuilder.appendLine("Time: %.2f ms".format(duration))
                    resultBuilder.appendLine("Sample: ${a[10]} + ${b[10]} = ${c[10]}")
                } else {
                    resultBuilder.appendLine("❌ Results verification failed!")
                }
                
                mlx.destroy()
            } else {
                resultBuilder.appendLine("❌ Failed to initialize MLX")
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error running demo", e)
            resultBuilder.appendLine("\n❌ Error: ${e.message}")
        }
        
        outputText.text = resultBuilder.toString()
    }
}
