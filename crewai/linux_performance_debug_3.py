# Importing Libraries
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
import os
import subprocess


# Ensure your OpenAI environment variables are set
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
        result = subprocess.run(
            ["top", "-bn1"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return f"Error collecting CPU stats: {result.stderr}"

        lines = result.stdout.split("\n")
        cpu_line = [line for line in lines if "Cpu(s)" in line or "%Cpu" in line]

        if cpu_line:
            return f"CPU Statistics:\n{cpu_line[0]}"
        return "CPU Statistics: Unable to parse CPU data"

    except Exception as e:
        return f"Error collecting CPU stats: {str(e)}"


@tool
def get_memory_usage() -> str:
    """
    Get current memory usage statistics from the Linux system.
    Returns total, used, free memory and swap information.
    """
    try:
        result = subprocess.run(
            ["free", "-h"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return f"Error collecting memory stats: {result.stderr}"

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
        iostat_result = subprocess.run(
            ["which", "iostat"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if iostat_result.returncode == 0:
            result = subprocess.run(
                ["iostat", "-x", "1", "2"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return f"Error collecting disk I/O stats: {result.stderr}"

            return f"Disk I/O Statistics:\n{result.stdout}"

        with open("/proc/diskstats", "r") as f:
            diskstats = f.read()

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
        result = subprocess.run(
            ["ip", "-s", "link"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            return f"Network Statistics:\n{result.stdout}"

        with open("/proc/net/dev", "r") as f:
            netdev = f.read()

        return f"Network Statistics (from /proc/net/dev):\n{netdev}"

    except Exception as e:
        return f"Error collecting network stats: {str(e)}"


# Define the Agents

monitor_agent = Agent(
    role="Linux System Monitor",
    goal="Collect comprehensive performance metrics from the current Linux system by executing system commands.",
    backstory=(
        "You are an expert Linux system administrator who is skilled at gathering "
        "real-time system metrics using standard Linux tools."
    ),
    tools=[get_cpu_utilization, get_memory_usage, get_disk_io, get_network_stats],
    allow_delegation=False,
    verbose=True
)

analyzer_agent = Agent(
    role="Linux Performance Analyzer",
    goal="Analyze collected Linux performance data to identify bottlenecks, anomalies, and high utilization.",
    backstory=(
        "You are a seasoned Linux performance engineer who can interpret CPU, memory, "
        "disk I/O, and network metrics and summarize likely issues."
    ),
    allow_delegation=False,
    verbose=True
)

diagnoser_agent = Agent(
    role="Linux Root Cause Diagnoser",
    goal="Determine probable root causes of Linux performance bottlenecks and recommend actionable solutions.",
    backstory=(
        "You are an experienced Linux troubleshooting specialist who can connect system symptoms "
        "to root causes and recommend practical remediation steps."
    ),
    allow_delegation=False,
    verbose=True
)


# Define Parallel Metric Collection Tasks
# These tasks are independent, so they can run in parallel.

collect_cpu_task = Task(
    description="""
        Use the get_cpu_utilization tool to collect current CPU statistics from the Linux system.
        Return only the collected CPU information in a clean format.
    """,
    expected_output="A report containing current CPU utilization statistics.",
    agent=monitor_agent,
    async_execution=True
)

collect_memory_task = Task(
    description="""
        Use the get_memory_usage tool to collect current memory and swap statistics from the Linux system.
        Return only the collected memory information in a clean format.
    """,
    expected_output="A report containing current memory and swap statistics.",
    agent=monitor_agent,
    async_execution=True
)

collect_disk_task = Task(
    description="""
        Use the get_disk_io tool to collect current disk I/O statistics from the Linux system.
        Return only the collected disk I/O information in a clean format.
    """,
    expected_output="A report containing current disk I/O statistics.",
    agent=monitor_agent,
    async_execution=True
)

collect_network_task = Task(
    description="""
        Use the get_network_stats tool to collect current network interface statistics from the Linux system.
        Return only the collected network information in a clean format.
    """,
    expected_output="A report containing current network interface statistics.",
    agent=monitor_agent,
    async_execution=True
)


# Define Dependent Tasks

analyze_performance_task = Task(
    description="""
        Review the CPU, memory, disk I/O, and network reports collected from the Linux system.

        Do the following:
        1. Identify any high utilization or unusual values
        2. Detect anomalies or signs of resource contention
        3. Highlight which subsystem appears to be under pressure
        4. Summarize the findings clearly
    """,
    expected_output="A concise analysis report highlighting performance issues and anomalies found in the collected metrics.",
    agent=analyzer_agent,
    context=[collect_cpu_task, collect_memory_task, collect_disk_task, collect_network_task]
)

diagnose_recommend_task = Task(
    description="""
        Based on the performance analysis report, determine the most probable root causes.

        Do the following:
        1. Identify the likely source of the problem
        2. Explain why that root cause is plausible
        3. Recommend specific remediation steps
        4. Include Linux commands or configuration suggestions where helpful
    """,
    expected_output="A diagnostic report with likely root causes and actionable recommendations.",
    agent=diagnoser_agent,
    context=[analyze_performance_task]
)


# Form the Crew
linux_debug_crew = Crew(
    agents=[monitor_agent, analyzer_agent, diagnoser_agent],
    tasks=[
        collect_cpu_task,
        collect_memory_task,
        collect_disk_task,
        collect_network_task,
        analyze_performance_task,
        diagnose_recommend_task
    ],
    process=Process.sequential,
    verbose=True
)


# Kick Off the Linux Performance Debugging Process
result_debug = linux_debug_crew.kickoff()

print("\n\n########################")
print("DEBUGGING PROCESS COMPLETE")
print("########################\n")
print(result_debug)
