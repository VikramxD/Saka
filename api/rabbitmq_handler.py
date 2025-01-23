"""RabbitMQ handler for managing video processing tasks.

This module provides a RabbitMQ client implementation for handling video processing tasks
in a distributed system. It manages connections, task publishing, and message consumption
with automatic reconnection and error handling capabilities.

Example:
    handler = RabbitMQHandler(settings)
    
    # Publishing a task
    task_id = handler.publish_task({
        "task_id": "123",
        "video_path": "/path/to/video.mp4",
        "parameters": {"calculate_ssim": True}
    })
    
    # Consuming tasks
    def process_task(task_data):
        # Process the video
        pass
    
    handler.consume_tasks(process_task)
"""

import pika
import json
from typing import Any, Dict, Callable, Optional
from loguru import logger
from configs.settings import RabbitMQSettings

class RabbitMQHandler:
    """Handles RabbitMQ connections and operations for video processing tasks.
    
    This class manages the connection to RabbitMQ, handles task publishing and consumption,
    and provides error handling and automatic reconnection capabilities.
    
    Attributes:
        settings (RabbitMQSettings): Configuration settings for RabbitMQ connection
        connection (pika.BlockingConnection): Active connection to RabbitMQ server
        channel (pika.Channel): Channel for communication with RabbitMQ
    
    Args:
        settings (RabbitMQSettings): Configuration settings for RabbitMQ
    
    Raises:
        pika.exceptions.AMQPConnectionError: If connection to RabbitMQ fails
    """
    
    def __init__(self, settings: RabbitMQSettings):
        self.settings = settings
        self.connection = None
        self.channel = None
        self._connect()
        
    def _connect(self):
        """Establishes connection to RabbitMQ and sets up exchanges and queues.
        
        This method:
        1. Creates a connection to RabbitMQ using the provided settings
        2. Declares exchanges and queues for video processing
        3. Sets up queue bindings and QoS parameters
        
        Raises:
            pika.exceptions.AMQPConnectionError: If connection fails
            pika.exceptions.AMQPChannelError: If channel operations fail
        """
        try:
            credentials = pika.PlainCredentials(
                self.settings.username,
                self.settings.password
            )
            parameters = pika.ConnectionParameters(
                host=self.settings.host,
                port=self.settings.port,
                virtual_host=self.settings.vhost,
                credentials=credentials,
                heartbeat=self.settings.heartbeat,
                connection_attempts=self.settings.connection_retry
            )
            
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            self.task_queue = f"{self.settings.queue_name}.tasks"
            self.result_queue = f"{self.settings.queue_name}.results"
            
            self.channel.queue_declare(queue=self.task_queue, durable=True)
            self.channel.queue_declare(queue=self.result_queue, durable=True)
            
            logger.info("Connected to RabbitMQ successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
            
    def publish_task(self, task_data: Dict[str, Any]) -> str:
        """Publishes a video processing task to RabbitMQ.
        
        Args:
            task_data (Dict[str, Any]): Task data containing:
                - task_id (str): Unique identifier for the task
                - video_path (str): Path to the video file
                - parameters (Dict): Processing parameters
        
        Returns:
            str: Task ID of the published task
        
        Raises:
            pika.exceptions.AMQPConnectionError: If connection is lost
            pika.exceptions.AMQPChannelError: If publishing fails
            
        Note:
            If connection is lost during publishing, the method will attempt
            to reconnect and retry the operation once.
        """
        try:
            message = json.dumps(task_data)
            self.channel.basic_publish(
                exchange='',
                routing_key=self.task_queue,
                body=message,
                properties=pika.BasicProperties(
                    delivery_mode=2,
                    correlation_id=task_data.get('task_id')
                )
            )
            logger.info(f"Published task: {task_data.get('task_id')}")
            return task_data.get('task_id')
            
        except pika.exceptions.AMQPConnectionError:
            logger.warning("Lost connection, attempting to reconnect...")
            self._connect()
            return self.publish_task(task_data)
        except Exception as e:
            logger.error(f"Failed to publish task: {e}")
            raise
            
    def consume_tasks(self, callback: Callable):
        """Starts consuming tasks from the queue.
        
        Sets up a consumer that processes incoming messages using the provided callback.
        The callback will be invoked for each message with the decoded task data.
        
        Args:
            callback (Callable): Function to process tasks, accepts task_data dict
        
        Note:
            - Messages are acknowledged only after successful processing
            - Failed messages are negatively acknowledged and requeued
            - The consumer runs indefinitely until interrupted
            
        Example:
            def process_task(task_data):
                # Process the video
                video_path = task_data['video_path']
                parameters = task_data['parameters']
                # ... process video ...
            
            handler.consume_tasks(process_task)
        """
        def wrapped_callback(ch, method, properties, body):
            try:
                task_data = json.loads(body)
                callback(task_data)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                logger.error(f"Error processing task: {e}")
                ch.basic_nack(
                    delivery_tag=method.delivery_tag,
                    requeue=True
                )
        
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue=self.task_queue,
            on_message_callback=wrapped_callback
        )
        
        logger.info(f"Started consuming from queue: {self.task_queue}")
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.channel.stop_consuming()
            
    def update_result(self, task_id: str, result: Dict[str, Any]):
        """Publish result to result queue."""
        try:
            message = json.dumps(result)
            self.channel.basic_publish(
                exchange='',
                routing_key=self.result_queue,
                body=message,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent
                    correlation_id=task_id
                )
            )
            logger.debug(f"Published result for task {task_id}: {result}")
        except Exception as e:
            logger.error(f"Failed to publish result: {e}")
            raise
            
    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task result from queue."""
        try:
            # Try to get a message from the result queue
            method_frame, properties, body = self.channel.basic_get(
                queue=self.result_queue,
                auto_ack=True
            )
            
            if method_frame:
                result = json.loads(body)
                if properties.correlation_id == task_id:
                    return result
                    
                # Put back if not our result
                self.channel.basic_publish(
                    exchange='',
                    routing_key=self.result_queue,
                    body=body,
                    properties=properties
                )
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to get result: {e}")
            raise
            
    def close(self):
        """Closes the RabbitMQ connection.
        
        This method should be called when the handler is no longer needed
        to ensure proper cleanup of resources.
        """
        if self.connection and not self.connection.is_closed:
            self.connection.close()
