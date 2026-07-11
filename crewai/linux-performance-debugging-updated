from crewai import Agent, Task, Crew
from crewai.tools import tool
import os
import subprocess

# Ensure your OpenAI environment variables are set
#openai_api_key = os.environ["OPENAI_API_KEY"]
#openai_base_url = os.environ["OPENAI_BASE_URL"]


@tool
def get_cpu_utilization() -> str:
    """
    Get the top CPU-consuming processes from the Linux system.
    Returns a per-process breakdown sorted by CPU% with PID, user, and command.
    """
    try:
        result = subprocess.run(
            ["ps", "aux", "--sort=-%cpu"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            return f"Top 20 CPU-Consuming Processes:\n{chr(10).join(lines[:21])}"

        return "Unable to collect CPU process data"

    except Exception as e:
        return f"Error collecting CPU stats: {str(e)}"


@tool
def get_memory_usage() -> str:
    """
    Get the top memory-consuming processes from the Linux system.
    Returns a per-process breakdown sorted by MEM% with PID, user, RSS, and command.
    """
    try:
        result = subprocess.run(
            ["ps", "aux", "--sort=-%mem"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            return f"Top 20 Memory-Consuming Processes:\n{chr(10).join(lines[:21])}"

        return "Unable to collect memory process data"

    except Exception as e:
        return f"Error collecting memory stats: {str(e)}"


@tool
def get_disk_io() -> str:
    """
    Get per-process disk I/O statistics from the Linux system.
    Returns read/write rates per process with PID and command name.
    Requires sysstat package (sudo apt install sysstat).
    """
    try:
        result = subprocess.run(
            ["pidstat", "-d", "1", "2"],
            capture_output=True, text=True, timeout=20
        )
        if result.returncode == 0:
            return f"Per-Process Disk I/O:\n{result.stdout.strip()}"

        return f"Error running pidstat: {result.stderr.strip()}"

    except FileNotFoundError:
        return "pidstat not found — install sysstat: sudo apt install sysstat"
    except Exception as e:
        return f"Error collecting disk I/O stats: {str(e)}"


@tool
def get_network_stats() -> str:
    """
    Get per-process network connection details from the Linux system.
    Returns every open TCP/UDP connection mapped to its owning process (PID, name, state).
    Uses ss which is part of iproute2 (pre-installed on most Linux distros).
    """
    try:
        result = subprocess.run(
            ["ss", "-tunap"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return f"Per-Process Network Connections:\n{result.stdout.strip()}"

        return f"Error running ss: {result.stderr.strip()}"

    except FileNotFoundError:
        return "ss not found — install iproute2: sudo apt install iproute2"
    except Exception as e:
        return f"Error collecting network stats: {str(e)}"


# ----- Agents -----

monitor_agent = Agent(
    role="Linux System Monitor",
    goal=(
        "Collect comprehensive, process-level performance metrics from the "
        "current Linux system by executing system commands. Always include "
        "the specific process names and PIDs responsible for resource usage."
    ),
    backstory=(
        "You are an expert Linux system administrator skilled at gathering "
        "real-time system metrics using standard Linux tools. You always dig "
        "down to the process level so the team knows exactly which processes "
        "are consuming resources."
    ),
    tools=[get_cpu_utilization, get_memory_usage, get_disk_io, get_network_stats],
    allow_delegation=False,
    verbose=True,
)

analyzer_agent = Agent(
    role="Linux Performance Analyzer",
    goal=(
        "Analyze collected Linux performance data to identify which specific "
        "processes are causing bottlenecks, anomalies, or high utilization in "
        "CPU, memory, disk I/O, and network."
    ),
    backstory=(
        "You are a seasoned Linux performance engineer. You read ps, top, "
        "pidstat, ss, and iostat output daily and can quickly spot the exact "
        "processes and PIDs behind any performance issue."
    ),
    allow_delegation=False,
    verbose=True,
)

diagnoser_agent = Agent(
    role="Linux Root Cause Diagnoser",
    goal=(
        "Determine the root causes of Linux performance problems, naming the "
        "exact processes and PIDs involved, and recommend actionable fixes."
    ),
    backstory=(
        "You are an experienced Linux troubleshooting specialist who connects "
        "system symptoms to root causes at the process level and recommends "
        "practical remediation steps, including specific commands."
    ),
    allow_delegation=False,
    verbose=True,
)

# ----- Tasks -----

collect_cpu_task = Task(
    description=(
        "Use the get_cpu_utilization tool to collect the top CPU-consuming "
        "processes. The output MUST list processes with their PIDs, user, "
        "CPU%, and command name."
    ),
    expected_output=(
        "A list of the top CPU-consuming processes (PID, user, CPU%, command)."
    ),
    agent=monitor_agent,
)

collect_memory_task = Task(
    description=(
        "Use the get_memory_usage tool to collect the top memory-consuming "
        "processes. The output MUST list processes with their PIDs, user, "
        "MEM%, RSS, and command name."
    ),
    expected_output=(
        "A list of the top memory-consuming processes (PID, user, MEM%, RSS, command)."
    ),
    agent=monitor_agent,
)

collect_disk_task = Task(
    description=(
        "Use the get_disk_io tool to collect per-process disk I/O statistics. "
        "The output MUST list processes with their PIDs, command names, and "
        "read/write rates (kB_rd/s, kB_wr/s)."
    ),
    expected_output=(
        "A per-process I/O breakdown (PID, command, read/write rates)."
    ),
    agent=monitor_agent,
)

collect_network_task = Task(
    description=(
        "Use the get_network_stats tool to collect per-process network "
        "connection details. The output MUST list each open connection with "
        "its owning PID, process name, local/remote address, and state."
    ),
    expected_output=(
        "A per-process network connection list (PID, process name, "
        "local/remote address, state)."
    ),
    agent=monitor_agent,
)

analyze_performance_task = Task(
    description=(
        "Review the CPU, memory, disk I/O, and network reports.\n\n"
        "Do the following:\n"
        "1. For EACH subsystem (CPU, memory, disk I/O, network), identify the "
        "specific processes (by name and PID) causing the highest utilization.\n"
        "2. Flag any process that is consuming an abnormal share of resources.\n"
        "3. Detect anomalies such as memory leaks, I/O storms, or SYN floods.\n"
        "4. Rank subsystems by severity of pressure.\n"
        "5. Produce a clear summary table: Process Name | PID | Resource | Usage."
    ),
    expected_output=(
        "A structured analysis report with a per-process summary table and "
        "ranked list of subsystems under pressure."
    ),
    agent=analyzer_agent,
    context=[collect_cpu_task, collect_memory_task, collect_disk_task, collect_network_task],
)

diagnose_recommend_task = Task(
    description=(
        "Based on the performance analysis report, determine root causes.\n\n"
        "Do the following:\n"
        "1. For each problematic process identified, explain the likely root "
        "cause (misconfiguration, resource leak, legitimate load, etc.).\n"
        "2. Provide specific remediation steps per process, including Linux "
        "commands to run (e.g., kill, renice, ionice, cgroups, ulimit, "
        "sysctl tunables).\n"
        "3. Suggest monitoring thresholds or alerts to prevent recurrence.\n"
        "4. Prioritize recommendations by impact."
    ),
    expected_output=(
        "A diagnostic report naming each offending process with its root cause "
        "and step-by-step remediation commands, prioritized by impact."
    ),
    agent=diagnoser_agent,
    context=[analyze_performance_task],
)

# ----- Crew -----

linux_debug_crew = Crew(
    agents=[monitor_agent, analyzer_agent, diagnoser_agent],
    tasks=[
        collect_cpu_task,
        collect_memory_task,
        collect_disk_task,
        collect_network_task,
        analyze_performance_task,
        diagnose_recommend_task,
    ],
    verbose=True
)

if __name__ == "__main__":
    try:
        result = linux_debug_crew.kickoff()
    except Exception as e:
        print(f"\nCrew execution failed: {e}")
        raise

    print("\n\n########################")
    print("DEBUGGING PROCESS COMPLETE")
    print("########################\n")
    print(result)
