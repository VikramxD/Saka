"""Task status callback implementation for video enhancer API."""

import time
import requests
import litserve as ls
from loguru import logger
from typing import Dict, Any, Optional, List


class TaskStatusCallback(ls.Callback):
    """Callback to notify webhook endpoints about task status changes."""
    
    def __init__(self):
        super().__init__()
        self.tasks = {}  # Store task info
    
    def _notify_webhook(self, task_id: str, status: Dict[str, Any]):
        """Send notification to webhook if configured."""
        task_info = self.tasks.get(task_id, {})
        webhook_url = task_info.get('webhook_url')
        
        if webhook_url:
            try:
                requests.post(webhook_url, json=status)
                logger.info(f"Notified webhook for task {task_id}")
            except Exception as e:
                logger.error(f"Failed to notify webhook for task {task_id}: {e}")
    
    def on_after_decode_request(self, lit_api, request: List[Dict[str, Any]], *args, **kwargs) -> None:
        """Store task info and notify webhook about task received."""
        # Extract task info from request
        if not request or not isinstance(request, list):
            return
            
        request_data = request[0]
        task_id = request_data.get('task_id')
        webhook_url = request_data.get('webhook_url')
        
        if task_id and webhook_url:
            self.tasks[task_id] = {
                'webhook_url': webhook_url,
                'start_time': time.time()
            }
            
            # Notify webhook about task received
            status = {
                'task_id': task_id,
                'status': 'received',
                'timestamp': time.time(),
                'message': 'Task received and queued for processing'
            }
            self._notify_webhook(task_id, status)
    
    def on_after_predict(self, lit_api, result: List[Dict[str, Any]], *args, **kwargs) -> None:
        """Notify webhook about task processing started."""
        if not result or not isinstance(result, list):
            return
            
        result_data = result[0]
        task_id = result_data.get('task_id')
        if task_id in self.tasks:
            status = {
                'task_id': task_id,
                'status': 'processing',
                'timestamp': time.time(),
                'message': 'Task processing started'
            }
            self._notify_webhook(task_id, status)
    
    def on_after_encode_response(self, lit_api, response: List[Dict[str, Any]], *args, **kwargs) -> None:
        """Notify webhook about task completion."""
        if not response or not isinstance(response, list):
            return
            
        response_data = response[0]
        task_id = response_data.get('task_id')
        if task_id in self.tasks:
            task_info = self.tasks[task_id]
            processing_time = time.time() - task_info['start_time']
            
            status = {
                'task_id': task_id,
                'status': response_data.get('status', 'unknown'),
                'timestamp': time.time(),
                'processing_time': processing_time,
                'message': response_data.get('message', 'Task completed'),
                'result': response_data
            }
            self._notify_webhook(task_id, status)
            
            # Cleanup task info
            del self.tasks[task_id]
