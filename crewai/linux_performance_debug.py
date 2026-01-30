# Importing Libraries
from crewai import Agent, Task, Crew
from crewai.tools import tool
import os
import subprocess


# Ensure your OpenAI API Key is set
openai_api_key = os.environ["OPENAI_API_KEY"]
openai_base_url = os.environ["OPENAI_BASE_URL"]



# Define Tools for Executing Linux Commands

@tool
def get_cpu_utilization() -> str:
    """
    Get current CPU utilization statistics from the Linux system.
    Returns CPU usage percentages including user, system, idle, and iowait.
    """
    try:
        # Using 'top' command with batch mode to get CPU stats
        result = subprocess.run(
            ['top', '-bn1'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Extract CPU line from top output
        lines = result.stdout.split('\n')
        cpu_line = [line for line in lines if 'Cpu(s)' in line or '%Cpu' in line]
        
        if cpu_line:
            return f"CPU Statistics:\n{cpu_line[0]}"
        else:
            return "CPU Statistics: Unable to parse CPU data"
            
    except Exception as e:
        return f"Error collecting CPU stats: {str(e)}"

@tool
def get_memory_usage() -> str:
    """
    Get current memory behavior statistics from the Linux system.
    This function focuses on memory pressure signals rather than swap usage,
    which is typically disabled in containerized environments.
    It captures runtime indicators such as runnable processes, free memory,
    and CPU wait time to help detect memory-related performance issues.
    """
    try:
        # Using 'free' command to get memory stats
        result = subprocess.run(
            ['vmstat', '1', '2'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        return f"Memory Statistics:\n{result.stdout}"
        
    except Exception as e:
        return f"Error collecting memory stats: {str(e)}"

@tool
def get_disk_io() -> str:
    """
    Get current disk I/O statistics from the Linux system.
    Returns read/write operations and throughput per disk device.
    """
    try:
        # Check if iostat is available, if not use vmstat
        iostat_result = subprocess.run(
            ['which', 'iostat'],
            capture_output=True,
            text=True
        )
        
        if iostat_result.returncode == 0:
            # iostat is available
            result = subprocess.run(
                ['iostat', '-x', '1', '2'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return f"Disk I/O Statistics:\n{result.stdout}"
        else:
            # Fallback to reading /proc/diskstats
            with open('/proc/diskstats', 'r') as f:
                lines = [l for l in f.readlines() if "loop" not in l][:25]
                diskstats = "".join(lines)
            return f"Disk I/O Statistics (from /proc/diskstats):\n{diskstats}"
            
    except Exception as e:
        return f"Error collecting disk I/O stats: {str(e)}"

@tool
def get_network_stats() -> str:
    """
    Get current network interface statistics from the Linux system.
    Returns packets sent/received, errors, and bandwidth usage per interface.
    """
    try:
        # Try using 'ip -s link' for network stats
        result = subprocess.run(
            ['ip', '-s', 'link'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            return f"Network Statistics:\n{result.stdout}"
        else:
            # Fallback to /proc/net/dev
            with open('/proc/net/dev', 'r') as f:
                netdev = f.read()
            return f"Network Statistics (from /proc/net/dev):\n{netdev}"
            
    except Exception as e:
        return f"Error collecting network stats: {str(e)}"

# Define the Agents with Tools

# Agent: SystemMonitorAgent (with tools to execute commands)
monitor_agent = Agent(
    role="Linux System Monitor",
    goal="Collect comprehensive performance metrics (CPU, Memory, I/O, Network) from the current Linux system by executing system commands.",
    backstory="You are an expert system administrator, adept at using various Linux commands and tools (like top, free, iostat, ip) to gather raw, detailed performance data. You have access to tools that execute these commands directly on the system.",
    tools=[get_cpu_utilization, get_memory_usage, get_disk_io, get_network_stats],
    allow_delegation=False,
    verbose=True
)

# Agent: PerformanceAnalyzerAgent
analyzer_agent = Agent(
    role="Linux Performance Analyzer",
    goal="Analyze collected Linux performance data to identify high utilization across CPU, Memory, I/O, and Network, and detect potential bottlenecks.",
    backstory="You are a seasoned performance engineer with a deep understanding of Linux system metrics. You can interpret raw data, identify trends, detect anomalies, and summarize key findings that indicate performance issues.",
    allow_delegation=False,
    verbose=True
)

# Agent: RootCauseDiagnoserAgent
diagnoser_agent = Agent(
    role="Linux Root Cause Diagnoser",
    goal="Determine the root cause of identified Linux performance bottlenecks and recommend actionable solutions to resolve them.",
    backstory="You are a highly experienced Linux troubleshooting specialist. Given detailed performance symptoms and analysis, you can deduce the underlying problems (e.g., specific processes, misconfigurations, hardware issues) and propose concrete, effective remediation strategies.",
    allow_delegation=False,
    verbose=True
)

# Define the Tasks

# Task: Collect Performance Metrics (using tools)
collect_metrics_task = Task(
    description="""
        Execute the available tools to collect real-time performance metrics from the Linux system:
        1. Use get_cpu_utilization tool to gather CPU statistics
        2. Use get_memory_usage tool to gather memory statistics
        3. Use get_disk_io tool to gather disk I/O statistics
        4. Use get_network_stats tool to gather network interface statistics
        5. Compile all the collected data into a comprehensive performance report
    """,
    expected_output="A detailed report containing actual CPU, Memory, Disk I/O, and Network utilization statistics collected from the system.",
    agent=monitor_agent,
)

# Task: Analyze Performance Data
analyze_performance_task = Task(
    description="""
        1. **Review Metrics**: Carefully examine the collected CPU, Memory, Disk I/O, and Network utilization report.
        2. **Identify High Utilization**: Pinpoint any components showing consistently high utilization or unusual spikes.
        3. **Detect Anomalies**: Look for any unexpected patterns, resource contention, or signs of performance degradation.
        4. **Summarize Findings**: Create a report detailing which resources are experiencing high utilization and any specific anomalies found.
    """,
    expected_output="A concise analysis report highlighting specific performance issues found in the collected metrics.",
    agent=analyzer_agent,
)

# Task: Diagnose Root Cause and Recommend Solutions
diagnose_recommend_task = Task(
    description="""
        1. **Diagnose Root Cause**: Based on the performance analysis report, identify the most probable root causes for the reported high utilization and performance issues.
        2. **Propose Solutions**: For each identified root cause, recommend specific, actionable steps or configuration changes to mitigate or resolve the performance problem.
    """,
    expected_output="A diagnostic report detailing potential root causes for the performance issues and a list of recommended solutions with specific commands or configurations to implement.",
    agent=diagnoser_agent,
    context=[collect_metrics_task, analyze_performance_task]
)

# Form the Crew for Linux Performance Debugging
linux_debug_crew = Crew(
    agents=[monitor_agent, analyzer_agent, diagnoser_agent],
    tasks=[collect_metrics_task, analyze_performance_task, diagnose_recommend_task],
    verbose=True,
)

# Kick Off the Linux Performance Debugging Process
# No simulated data needed - agents will execute real commands
result_debug = linux_debug_crew.kickoff()

print("\n\n########################")
print("DEBUGGING PROCESS COMPLETE")
print("########################\n")
print(result_debug)
