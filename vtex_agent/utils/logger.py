"""Agent-specific logging utilities."""
import logging
import os
from pathlib import Path
from typing import Optional


def get_agent_logger(agent_name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for a specific agent.
    
    Logs are written to both:
    - Console (INFO level and above)
    - File (all levels, append mode)
    
    Args:
        agent_name: Name of the agent (e.g., 'legacy_site_agent')
        log_dir: Directory for log files (default: logs/ in project root)
        
    Returns:
        Configured logger instance
    """
    # Determine log directory
    if log_dir is None:
        # Get project root (parent of vtex_agent)
        project_root = Path(__file__).parent.parent.parent
        log_dir = project_root / "logs"
    else:
        log_dir = Path(log_dir)
    
    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(f"vtex_agent.{agent_name}")
    logger.setLevel(logging.DEBUG)
    
    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger
    
    # File handler (append mode)
    log_file = log_dir / f"{agent_name}_log.txt"
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

