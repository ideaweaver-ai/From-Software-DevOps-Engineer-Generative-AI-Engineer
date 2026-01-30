# Importing Libraries
from crewai import Agent, Task, Crew
from crewai.tools import tool
import os
import subprocess


# -----------------------------
# Env checks (don't crash hard)
# -----------------------------
openai_api_key = os.getenv("OPENAI_API_KEY", "")
openai_base_url = os.getenv("OPENAI_BASE_URL", "")

if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
# OPENAI_BASE_URL may or may not be required in your setup; keep it optional.


# ----------------------------------------
# Tools for Executing Linux Commands
# ----------------------------------------

@tool
def get_cpu_utilization() -> str:
    """
    Get current CPU utilization statistics from the Linux system.
    Returns CPU usage percentages including user, system, idle, and iowait.
    """
    try:
        result = subprocess.run(
            ['top', '-bn1'],
            capture_output=True,
            text=True,
            timeout=10
        )

        lines = result.stdout.split('\n')
        cpu_line = [line for line in lines if ('Cpu(s)' in line or '%Cpu' in line)]

        if cpu_line:
            return f"CPU Statistics:\n{cpu_line[0]}"
        return "CPU Statistics: Unable to parse CPU data"

    except Exception as e:
        return f"Error collecting CPU stats: {str(e)}"


@tool
def get_top_cpu_processes() -> str:
    """
    Pinpoint the exact processes consuming the most CPU RIGHT NOW.
    Uses top in batch mode sorted by %CPU and returns the top entries.
    """
    try:
        # top output contains header + process table
        # We'll keep it simple: return the top 15 lines of the process section.
        result = subprocess.run(
            ['top', '-b', '-n', '1', '-o', '%CPU'],
            capture_output=True,
            text=True,
            timeout=10
        )

        out = result.stdout.strip().splitlines()

        # Find the process table header line that starts with "PID"
        pid_header_idx = None
        for i, line in enumerate(out):
            if line.strip().startswith("PID "):
                pid_header_idx = i
                break

        if pid_header_idx is None:
            # Fallback: just return some lines
            return "Top CPU Processes (raw top output):\n" + "\n".join(out[:40])

        header = out[pid_header_idx]
        # Return header + next 15 process lines
        proc_lines = out[pid_header_idx: pid_header_idx + 16]

        return "Top CPU Processes (sorted by %CPU):\n" + "\n".join(proc_lines)

    except Exception as e:
        return f"Error collecting top CPU processes: {str(e)}"


@tool
def get_memory_usage() -> str:
    """
    Get current memory usage statistics from the Linux system.
    Returns total, used, free memory and swap information.
    """
    try:
        result = subprocess.run(
            ['free', '-h'],
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
        iostat_result = subprocess.run(['which', 'iostat'], capture_output=True, text=True)

        if iostat_result.returncode == 0:
            result = subprocess.run(
                ['iostat', '-x', '1', '2'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return f"Disk I/O Statistics:\n{result.stdout}"

        # Fallback: /proc/diskstats
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
        result = subprocess.run(
            ['ip', '-s', 'link'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            return f"Network Statistics:\n{result.stdout}"

        # Fallback to /proc/net/dev
        with open('/proc/net/dev', 'r') as f:
            netdev = f.read()
        return f"Network Statistics (from /proc/net/dev):\n{netdev}"

    except Exception as e:
        return f"Error collecting network stats: {str(e)}"


# ----------------------------------------
# Agents
# ----------------------------------------

monitor_agent = Agent(
    role="Linux System Monitor",
    goal=(
        "Collect comprehensive performance metrics (CPU, top CPU processes, Memory, I/O, Network) "
        "from the current Linux system by executing system commands."
    ),
    backstory=(
        "You are an expert system administrator. You collect raw performance data using Linux tools "
        "and produce a clean snapshot report that includes both CPU summary and the top CPU-consuming processes."
    ),
    tools=[
        get_cpu_utilization,
        get_top_cpu_processes,   # ✅ NEW TOOL ADDED
        get_memory_usage,
        get_disk_io,
        get_network_stats
    ],
    allow_delegation=False,
    verbose=True
)

analyzer_agent = Agent(
    role="Linux Performance Analyzer",
    goal="Analyze collected Linux performance data to identify high utilization across CPU, Memory, I/O, and Network, and detect potential bottlenecks.",
    backstory="You are a seasoned performance engineer. You interpret the metrics and identify what is high and why, including which process is driving CPU load.",
    allow_delegation=False,
    verbose=True
)

diagnoser_agent = Agent(
    role="Linux Root Cause Diagnoser",
    goal="Determine the root cause of identified Linux performance bottlenecks and recommend actionable solutions to resolve them.",
    backstory="You are a Linux troubleshooting specialist. You identify the most likely root cause and provide practical next steps and commands.",
    allow_delegation=False,
    verbose=True
)


# ----------------------------------------
# Tasks
# ----------------------------------------

collect_metrics_task = Task(
    description="""
Execute the available tools to collect real-time performance metrics from the Linux system:

1. Use get_cpu_utilization to gather overall CPU summary
2. Use get_top_cpu_processes to pinpoint which process(es) are consuming CPU
3. Use get_memory_usage to gather memory statistics
4. Use get_disk_io to gather disk I/O statistics
5. Use get_network_stats to gather network interface statistics

Compile all collected data into a comprehensive performance report.
""",
    expected_output="A detailed report containing CPU summary, top CPU-consuming processes, Memory, Disk I/O, and Network statistics.",
    agent=monitor_agent,
)

analyze_performance_task = Task(
    description="""
1. Review the collected report (CPU, Top CPU processes, Memory, Disk I/O, Network).
2. Identify high utilization or anomalies.
3. If CPU is high, explicitly name the top offending process(es) with PID and %CPU.
4. Summarize findings.
""",
    expected_output="A concise analysis report highlighting performance issues and the specific top CPU process(es) if CPU is high.",
    agent=analyzer_agent,
)

diagnose_recommend_task = Task(
    description="""
1. Based on the performance analysis, identify probable root causes.
2. Propose actionable solutions.
3. If CPU is the bottleneck, give next-step commands focusing on the top CPU PID(s)
   (e.g., pidstat/perf/strace/renice/cpulimit/cgroups).
""",
    expected_output="A diagnostic report detailing root causes and recommended solutions with concrete commands.",
    agent=diagnoser_agent,
    context=[collect_metrics_task, analyze_performance_task]
)


# ----------------------------------------
# Crew
# ----------------------------------------

linux_debug_crew = Crew(
    agents=[monitor_agent, analyzer_agent, diagnoser_agent],
    tasks=[collect_metrics_task, analyze_performance_task, diagnose_recommend_task],
    verbose=True,
)

# Kick off
result_debug = linux_debug_crew.kickoff()

print("\n\n########################")
print("DEBUGGING PROCESS COMPLETE")
print("########################\n")
print(result_debug)
