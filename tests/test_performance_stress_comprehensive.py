"""Comprehensive performance and stress tests for the keylogger system.
This module contains tests for:
- Performance benchmarking and profiling
- Resource usage monitoring and limits
- Stress testing under high load
- Concurrency and threading performance
- Memory leak detection
- CPU and I/O intensive scenarios
- System resource exhaustion handling
"""

import unittest
import time
import threading
import multiprocessing
import psutil
import gc
import os
import tempfile
import shutil
import json
import statistics
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from typing import List, Dict, Any, Callable

# Import system components
from core.keylogger import KeyloggerCore
from core.config_manager import ConfigManager
from core.encryption_manager import EncryptionManager
from core.logging_manager import LoggingManager
from utils.performance_monitor import PerformanceMonitor
from listeners.keyboard_listener import KeyboardListener
from listeners.mouse_listener import MouseListener
from listeners.clipboard_listener import ClipboardListener


class PerformanceBenchmark:
    """Performance benchmarking utility."""
    
    def __init__(self):
        self.results = []
        self.start_time = None
        self.end_time = None
        self.process = psutil.Process()
        self.initial_memory = None
        self.peak_memory = 0
        self.initial_cpu_time = None
    
    def start_benchmark(self, test_name: str):
        """Start performance benchmark."""
        gc.collect()  # Clean up before benchmark
        self.start_time = time.perf_counter()
        self.initial_memory = self.process.memory_info().rss
        self.initial_cpu_time = self.process.cpu_times()
        self.peak_memory = self.initial_memory
        self.test_name = test_name
    
    def end_benchmark(self) -> Dict[str, Any]:
        """End benchmark and return results."""
        self.end_time = time.perf_counter()
        final_memory = self.process.memory_info().rss
        final_cpu_time = self.process.cpu_times()
        
        result = {
            'test_name': self.test_name,
            'duration_seconds': self.end_time - self.start_time,
            'memory_initial_mb': self.initial_memory / (1024**2),
            'memory_final_mb': final_memory / (1024**2),
            'memory_peak_mb': self.peak_memory / (1024**2),
            'memory_delta_mb': (final_memory - self.initial_memory) / (1024**2),
            'cpu_user_time': final_cpu_time.user - self.initial_cpu_time.user,
            'cpu_system_time': final_cpu_time.system - self.initial_cpu_time.system,
            'timestamp': time.time()
        }
        
        self.results.append(result)
        return result
    
    def update_peak_memory(self):
        """Update peak memory usage."""
        current_memory = self.process.memory_info().rss
        self.peak_memory = max(self.peak_memory, current_memory)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary statistics."""
        if not self.results:
            return {}
        
        durations = [r['duration_seconds'] for r in self.results]
        memory_deltas = [r['memory_delta_mb'] for r in self.results]
        
        return {
            'total_tests': len(self.results),
            'total_duration': sum(durations),
            'avg_duration': statistics.mean(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'avg_memory_delta': statistics.mean(memory_deltas),
            'max_memory_delta': max(memory_deltas),
            'results': self.results
        }


class StressTestRunner:
    """Stress test execution utility."""
    
    def __init__(self):
        self.active_threads = []
        self.errors = []
        self.results = []
        self.stop_event = threading.Event()
    
    def run_concurrent_stress_test(self, test_func: Callable, 
                                 thread_count: int, 
                                 duration_seconds: int,
                                 *args, **kwargs) -> Dict[str, Any]:
        """Run stress test with multiple concurrent threads."""
        self.errors.clear()
        self.results.clear()
        self.stop_event.clear()
        
        start_time = time.perf_counter()
        
        # Start worker threads
        for i in range(thread_count):
            thread = threading.Thread(
                target=self._stress_worker,
                args=(test_func, i, *args),
                kwargs=kwargs,
                daemon=True
            )
            thread.start()
            self.active_threads.append(thread)
        
        # Run for specified duration
        time.sleep(duration_seconds)
        self.stop_event.set()
        
        # Wait for threads to complete
        for thread in self.active_threads:
            thread.join(timeout=5.0)
        
        end_time = time.perf_counter()
        
        return {
            'duration': end_time - start_time,
            'thread_count': thread_count,
            'total_operations': len(self.results),
            'operations_per_second': len(self.results) / (end_time - start_time),
            'error_count': len(self.errors),
            'error_rate': len(self.errors) / max(1, len(self.results)),
            'errors': self.errors[:10]  # First 10 errors
        }
    
    def _stress_worker(self, test_func: Callable, worker_id: int, *args, **kwargs):
        """Worker thread for stress testing."""
        operation_count = 0
        
        while not self.stop_event.is_set():
            try:
                result = test_func(worker_id, operation_count, *args, **kwargs)
                self.results.append({
                    'worker_id': worker_id,
                    'operation_count': operation_count,
                    'result': result,
                    'timestamp': time.time()
                })
                operation_count += 1
                
            except Exception as e:
                self.errors.append({
                    'worker_id': worker_id,
                    'operation_count': operation_count,
                    'error': str(e),
                    'timestamp': time.time()
                })
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.001)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'config.json')
        self.key_file = os.path.join(self.test_dir, 'key.key')
        self.benchmark = PerformanceBenchmark()
        
        # Create test configuration
        config_data = {
            'logging': {'log_file': os.path.join(self.test_dir, 'test.log')},
            'encryption': {'key_file': self.key_file},
            'performance': {
                'monitor_interval': 1.0,
                'history_size': 50,
                'cpu_alert_threshold': 90.0,
                'memory_alert_threshold': 90.0
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_keylogger_initialization_performance(self):
        """Benchmark keylogger initialization performance."""
        self.benchmark.start_benchmark('keylogger_initialization')
        
        # Initialize keylogger multiple times
        for i in range(10):
            keylogger = KeyloggerCore(self.config_file)
            self.benchmark.update_peak_memory()
            del keylogger
            gc.collect()
        
        result = self.benchmark.end_benchmark()
        
        # Performance assertions
        self.assertLess(result['duration_seconds'], 5.0, "Initialization too slow")
        self.assertLess(result['memory_delta_mb'], 50.0, "Memory usage too high")
    
    def test_encryption_performance_benchmark(self):
        """Benchmark encryption/decryption performance."""
        encryption_manager = EncryptionManager(self.key_file)
        
        self.benchmark.start_benchmark('encryption_performance')
        
        # Test data of various sizes
        test_data_sizes = [100, 1000, 10000, 100000]  # bytes
        
        for size in test_data_sizes:
            test_data = 'A' * size
            
            # Benchmark encryption
            for _ in range(100):
                encrypted = encryption_manager.encrypt_data(test_data)
                decrypted = encryption_manager.decrypt_data(encrypted)
                self.assertEqual(decrypted, test_data)
                self.benchmark.update_peak_memory()
        
        result = self.benchmark.end_benchmark()
        
        # Performance assertions
        self.assertLess(result['duration_seconds'], 10.0, "Encryption too slow")
        self.assertLess(result['memory_delta_mb'], 100.0, "Memory usage too high")
    
    def test_logging_performance_benchmark(self):
        """Benchmark logging performance."""
        config = ConfigManager(self.config_file)
        encryption = EncryptionManager(self.key_file)
        logging_manager = LoggingManager(config, encryption)
        
        self.benchmark.start_benchmark('logging_performance')
        
        try:
            logging_manager.start()
            
            # Generate high volume of log entries
            for i in range(1000):
                from core.logging_manager import LogEntry
                entry = LogEntry(
                    event_type="benchmark",
                    content=f"Benchmark log entry {i} with some content"
                )
                logging_manager.log_event(entry)
                
                if i % 100 == 0:
                    self.benchmark.update_peak_memory()
            
            # Flush all entries
            logging_manager.flush_buffer()
            
        finally:
            logging_manager.stop()
        
        result = self.benchmark.end_benchmark()
        
        # Performance assertions
        self.assertLess(result['duration_seconds'], 15.0, "Logging too slow")
        self.assertLess(result['memory_delta_mb'], 200.0, "Memory usage too high")
    
    def test_performance_monitor_overhead(self):
        """Test performance monitor overhead."""
        keylogger = KeyloggerCore(self.config_file)
        
        self.benchmark.start_benchmark('performance_monitor_overhead')
        
        # Run with performance monitoring
        performance_monitor = PerformanceMonitor(keylogger)
        
        # Simulate monitoring for a period
        for _ in range(10):
            performance_monitor._collect_system_metrics()
            performance_monitor._collect_keylogger_metrics()
            performance_monitor._check_performance_alerts()
            self.benchmark.update_peak_memory()
            time.sleep(0.1)
        
        result = self.benchmark.end_benchmark()
        
        # Performance assertions
        self.assertLess(result['duration_seconds'], 5.0, "Monitor overhead too high")
        self.assertLess(result['memory_delta_mb'], 30.0, "Monitor memory usage too high")


class TestStressScenarios(unittest.TestCase):
    """Stress testing scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'config.json')
        self.key_file = os.path.join(self.test_dir, 'key.key')
        self.stress_runner = StressTestRunner()
        
        # Create test configuration
        config_data = {
            'logging': {'log_file': os.path.join(self.test_dir, 'test.log')},
            'encryption': {'key_file': self.key_file}
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _encryption_stress_worker(self, worker_id: int, operation_count: int, 
                                encryption_manager: EncryptionManager) -> bool:
        """Worker function for encryption stress testing."""
        test_data = f"Worker {worker_id} operation {operation_count} data"
        encrypted = encryption_manager.encrypt_data(test_data)
        decrypted = encryption_manager.decrypt_data(encrypted)
        return decrypted == test_data
    
    def test_concurrent_encryption_stress(self):
        """Stress test concurrent encryption operations."""
        encryption_manager = EncryptionManager(self.key_file)
        
        # Run stress test with multiple threads
        result = self.stress_runner.run_concurrent_stress_test(
            self._encryption_stress_worker,
            thread_count=10,
            duration_seconds=5,
            encryption_manager=encryption_manager
        )
        
        # Verify stress test results
        self.assertGreater(result['total_operations'], 100, "Too few operations completed")
        self.assertLess(result['error_rate'], 0.01, "Error rate too high")
        self.assertGreater(result['operations_per_second'], 10, "Throughput too low")
    
    def _logging_stress_worker(self, worker_id: int, operation_count: int,
                             logging_manager: LoggingManager) -> bool:
        """Worker function for logging stress testing."""
        from core.logging_manager import LogEntry
        entry = LogEntry(
            event_type="stress_test",
            content=f"Worker {worker_id} stress operation {operation_count}"
        )
        logging_manager.log_event(entry)
        return True
    
    def test_concurrent_logging_stress(self):
        """Stress test concurrent logging operations."""
        config = ConfigManager(self.config_file)
        encryption = EncryptionManager(self.key_file)
        logging_manager = LoggingManager(config, encryption)
        
        try:
            logging_manager.start()
            
            # Run stress test
            result = self.stress_runner.run_concurrent_stress_test(
                self._logging_stress_worker,
                thread_count=8,
                duration_seconds=3,
                logging_manager=logging_manager
            )
            
            # Flush remaining entries
            logging_manager.flush_buffer()
            
            # Verify results
            self.assertGreater(result['total_operations'], 50, "Too few operations completed")
            self.assertLess(result['error_rate'], 0.05, "Error rate too high")
            
        finally:
            logging_manager.stop()
    
    def _keylogger_stress_worker(self, worker_id: int, operation_count: int,
                               keylogger: KeyloggerCore) -> bool:
        """Worker function for keylogger stress testing."""
        # Simulate various keylogger operations
        keylogger.log_event('stress_test', f'Worker {worker_id} event {operation_count}')
        
        # Simulate getting stats (common operation)
        stats = keylogger.get_stats()
        return 'total_events' in stats
    
    def test_keylogger_core_stress(self):
        """Stress test keylogger core operations."""
        keylogger = KeyloggerCore(self.config_file)
        
        try:
            keylogger.start()
            
            # Run stress test
            result = self.stress_runner.run_concurrent_stress_test(
                self._keylogger_stress_worker,
                thread_count=6,
                duration_seconds=4,
                keylogger=keylogger
            )
            
            # Verify results
            self.assertGreater(result['total_operations'], 30, "Too few operations completed")
            self.assertLess(result['error_rate'], 0.1, "Error rate too high")
            
        finally:
            keylogger.stop()


class TestResourceLimits(unittest.TestCase):
    """Test system resource limits and exhaustion scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'config.json')
        self.key_file = os.path.join(self.test_dir, 'key.key')
        
        # Create test configuration
        config_data = {
            'logging': {'log_file': os.path.join(self.test_dir, 'test.log')},
            'encryption': {'key_file': self.key_file}
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_memory_usage_limits(self):
        """Test memory usage stays within reasonable limits."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        keylogger = KeyloggerCore(self.config_file)
        
        try:
            keylogger.start()
            
            # Generate significant load
            for i in range(1000):
                keylogger.log_event('memory_test', f'Event {i} with some data')
                
                # Check memory every 100 events
                if i % 100 == 0:
                    current_memory = process.memory_info().rss
                    memory_increase = (current_memory - initial_memory) / (1024**2)
                    
                    # Memory should not increase excessively
                    self.assertLess(memory_increase, 500, 
                                  f"Memory usage too high: {memory_increase:.1f}MB")
            
            # Final memory check
            final_memory = process.memory_info().rss
            total_increase = (final_memory - initial_memory) / (1024**2)
            self.assertLess(total_increase, 200, 
                          f"Total memory increase too high: {total_increase:.1f}MB")
            
        finally:
            keylogger.stop()
    
    def test_file_descriptor_limits(self):
        """Test file descriptor usage stays within limits."""
        process = psutil.Process()
        
        # Get initial file descriptor count
        try:
            if hasattr(process, 'num_fds'):
                initial_fds = process.num_fds()
            else:
                initial_fds = process.num_handles()
        except (AttributeError, psutil.AccessDenied):
            self.skipTest("Cannot access file descriptor information")
        
        keylogger = KeyloggerCore(self.config_file)
        
        try:
            keylogger.start()
            
            # Perform operations that might open files
            for i in range(100):
                keylogger.log_event('fd_test', f'Event {i}')
                
                if i % 20 == 0:
                    # Check file descriptor count
                    try:
                        if hasattr(process, 'num_fds'):
                            current_fds = process.num_fds()
                        else:
                            current_fds = process.num_handles()
                        
                        fd_increase = current_fds - initial_fds
                        self.assertLess(fd_increase, 50, 
                                      f"Too many file descriptors opened: {fd_increase}")
                    except (AttributeError, psutil.AccessDenied):
                        pass
            
        finally:
            keylogger.stop()
    
    def test_cpu_usage_under_load(self):
        """Test CPU usage under high load scenarios."""
        keylogger = KeyloggerCore(self.config_file)
        process = psutil.Process()
        
        try:
            keylogger.start()
            
            # Generate high load
            start_time = time.time()
            cpu_samples = []
            
            while time.time() - start_time < 5.0:  # Run for 5 seconds
                # Generate events rapidly
                for i in range(50):
                    keylogger.log_event('cpu_test', f'High frequency event {i}')
                
                # Sample CPU usage
                cpu_percent = process.cpu_percent()
                if cpu_percent > 0:  # Only record non-zero samples
                    cpu_samples.append(cpu_percent)
                
                time.sleep(0.1)
            
            # Analyze CPU usage
            if cpu_samples:
                avg_cpu = statistics.mean(cpu_samples)
                max_cpu = max(cpu_samples)
                
                # CPU usage should be reasonable
                self.assertLess(avg_cpu, 50.0, f"Average CPU usage too high: {avg_cpu:.1f}%")
                self.assertLess(max_cpu, 80.0, f"Peak CPU usage too high: {max_cpu:.1f}%")
            
        finally:
            keylogger.stop()


class TestMemoryLeakDetection(unittest.TestCase):
    """Test for memory leaks in long-running scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'config.json')
        self.key_file = os.path.join(self.test_dir, 'key.key')
        
        # Create test configuration
        config_data = {
            'logging': {'log_file': os.path.join(self.test_dir, 'test.log')},
            'encryption': {'key_file': self.key_file}
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_encryption_memory_leak(self):
        """Test for memory leaks in encryption operations."""
        encryption_manager = EncryptionManager(self.key_file)
        process = psutil.Process()
        
        # Baseline memory measurement
        gc.collect()
        baseline_memory = process.memory_info().rss
        memory_samples = []
        
        # Perform many encryption operations
        for cycle in range(10):
            # Perform operations
            for i in range(100):
                test_data = f"Memory leak test data {cycle}-{i}" * 10
                encrypted = encryption_manager.encrypt_data(test_data)
                decrypted = encryption_manager.decrypt_data(encrypted)
                self.assertEqual(decrypted, test_data)
            
            # Force garbage collection and measure memory
            gc.collect()
            current_memory = process.memory_info().rss
            memory_increase = (current_memory - baseline_memory) / (1024**2)
            memory_samples.append(memory_increase)
        
        # Analyze memory growth trend
        if len(memory_samples) >= 5:
            # Check if memory is consistently growing
            recent_avg = statistics.mean(memory_samples[-3:])
            early_avg = statistics.mean(memory_samples[:3])
            
            memory_growth = recent_avg - early_avg
            self.assertLess(memory_growth, 20.0, 
                          f"Potential memory leak detected: {memory_growth:.1f}MB growth")
    
    def test_logging_memory_leak(self):
        """Test for memory leaks in logging operations."""
        config = ConfigManager(self.config_file)
        encryption = EncryptionManager(self.key_file)
        logging_manager = LoggingManager(config, encryption)
        process = psutil.Process()
        
        try:
            logging_manager.start()
            
            # Baseline memory measurement
            gc.collect()
            baseline_memory = process.memory_info().rss
            memory_samples = []
            
            # Perform many logging operations
            for cycle in range(8):
                # Generate log entries
                for i in range(200):
                    from core.logging_manager import LogEntry
                    entry = LogEntry(
                        event_type="memory_leak_test",
                        content=f"Cycle {cycle} entry {i} with test data" * 5
                    )
                    logging_manager.log_event(entry)
                
                # Flush and measure memory
                logging_manager.flush_buffer()
                gc.collect()
                
                current_memory = process.memory_info().rss
                memory_increase = (current_memory - baseline_memory) / (1024**2)
                memory_samples.append(memory_increase)
            
            # Analyze memory growth
            if len(memory_samples) >= 4:
                recent_avg = statistics.mean(memory_samples[-2:])
                early_avg = statistics.mean(memory_samples[:2])
                
                memory_growth = recent_avg - early_avg
                self.assertLess(memory_growth, 30.0,
                              f"Potential memory leak detected: {memory_growth:.1f}MB growth")
            
        finally:
            logging_manager.stop()


class TestConcurrencyPerformance(unittest.TestCase):
    """Test performance under various concurrency scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'config.json')
        self.key_file = os.path.join(self.test_dir, 'key.key')
        
        # Create test configuration
        config_data = {
            'logging': {'log_file': os.path.join(self.test_dir, 'test.log')},
            'encryption': {'key_file': self.key_file}
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_thread_scaling_performance(self):
        """Test performance scaling with different thread counts."""
        encryption_manager = EncryptionManager(self.key_file)
        
        def encryption_task():
            """Single encryption task."""
            test_data = "Thread scaling test data" * 10
            encrypted = encryption_manager.encrypt_data(test_data)
            decrypted = encryption_manager.decrypt_data(encrypted)
            return decrypted == test_data
        
        # Test with different thread counts
        thread_counts = [1, 2, 4, 8]
        results = {}
        
        for thread_count in thread_counts:
            start_time = time.perf_counter()
            
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                # Submit tasks
                futures = [executor.submit(encryption_task) for _ in range(100)]
                
                # Wait for completion
                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    self.assertTrue(result)
                    completed += 1
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            throughput = 100 / duration
            
            results[thread_count] = {
                'duration': duration,
                'throughput': throughput,
                'completed': completed
            }
        
        # Analyze scaling efficiency
        single_thread_throughput = results[1]['throughput']
        
        for thread_count in thread_counts[1:]:
            current_throughput = results[thread_count]['throughput']
            scaling_factor = current_throughput / single_thread_throughput
            
            # Should show some improvement with more threads
            if thread_count <= multiprocessing.cpu_count():
                self.assertGreater(scaling_factor, 1.2, 
                                 f"Poor scaling with {thread_count} threads")
    
    def test_contention_handling(self):
        """Test handling of resource contention scenarios."""
        config = ConfigManager(self.config_file)
        encryption = EncryptionManager(self.key_file)
        logging_manager = LoggingManager(config, encryption)
        
        try:
            logging_manager.start()
            
            # Create high contention scenario
            def contention_worker(worker_id):
                """Worker that creates resource contention."""
                operations = 0
                errors = 0
                
                for i in range(50):
                    try:
                        # Multiple operations that might contend
                        from core.logging_manager import LogEntry
                        entry = LogEntry(
                            event_type="contention_test",
                            content=f"Worker {worker_id} operation {i}"
                        )
                        logging_manager.log_event(entry)
                        
                        # Also test encryption contention
                        test_data = f"Worker {worker_id} data {i}"
                        encrypted = encryption.encrypt_data(test_data)
                        decrypted = encryption.decrypt_data(encrypted)
                        
                        if decrypted == test_data:
                            operations += 1
                        
                    except Exception:
                        errors += 1
                
                return {'operations': operations, 'errors': errors}
            
            # Run contention test
            start_time = time.perf_counter()
            
            with ThreadPoolExecutor(max_workers=12) as executor:
                futures = [executor.submit(contention_worker, i) for i in range(12)]
                
                total_operations = 0
                total_errors = 0
                
                for future in as_completed(futures):
                    result = future.result()
                    total_operations += result['operations']
                    total_errors += result['errors']
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Verify contention handling
            error_rate = total_errors / max(1, total_operations + total_errors)
            throughput = total_operations / duration
            
            self.assertLess(error_rate, 0.1, f"High error rate under contention: {error_rate:.2%}")
            self.assertGreater(throughput, 10, f"Low throughput under contention: {throughput:.1f} ops/sec")
            
        finally:
            logging_manager.stop()


if __name__ == '__main__':
    # Create test suite with performance focus
    suite = unittest.TestSuite()
    
    # Add benchmark tests
    suite.addTest(unittest.makeSuite(TestPerformanceBenchmarks))
    
    # Add stress tests
    suite.addTest(unittest.makeSuite(TestStressScenarios))
    
    # Add resource limit tests
    suite.addTest(unittest.makeSuite(TestResourceLimits))
    
    # Add memory leak tests
    suite.addTest(unittest.makeSuite(TestMemoryLeakDetection))
    
    # Add concurrency tests
    suite.addTest(unittest.makeSuite(TestConcurrencyPerformance))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"PERFORMANCE TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\n')[0]}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\n')[-2]}")