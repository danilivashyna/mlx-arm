package com.mlxarm.demo

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.mlxarm.demo.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Initialize MLX
        val mlx = MLXContext()
        
        // Get device info
        val deviceInfo = mlx.getDeviceInfo()
        
        // Display info
        binding.textView.text = """
            MLX-ARM Demo
            
            Device Info:
            ${deviceInfo.cpuName}
            ${deviceInfo.gpuName}
            Vulkan: ${deviceInfo.vulkanVersion}
            
            Status: ${if (deviceInfo.isInitialized) "✓ Ready" else "✗ Failed"}
        """.trimIndent()
        
        // Run vector add test
        binding.buttonRunTest.setOnClickListener {
            runVectorAddTest()
        }
    }
    
    private fun runVectorAddTest() {
        val mlx = MLXContext()
        
        // Create test vectors
        val a = FloatArray(1024) { it.toFloat() }
        val b = FloatArray(1024) { it * 2.0f }
        
        // Run on GPU
        val startTime = System.nanoTime()
        val result = mlx.vectorAdd(a, b)
        val endTime = System.nanoTime()
        
        val timeMs = (endTime - startTime) / 1_000_000.0
        
        // Verify first few results
        val isCorrect = (0 until 5).all { i ->
            Math.abs(result[i] - (a[i] + b[i])) < 0.001f
        }
        
        binding.textResult.text = """
            Vector Add Test (1024 elements)
            
            Time: %.2f ms
            Result: ${if (isCorrect) "✓ PASS" else "✗ FAIL"}
            
            First 5 results:
            ${result.take(5).joinToString(", ") { "%.1f".format(it) }}
        """.trimIndent().format(timeMs)
    }

    companion object {
        init {
            System.loadLibrary("mlx-core")
        }
    }
}
