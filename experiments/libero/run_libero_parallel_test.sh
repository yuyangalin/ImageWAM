#!/bin/bash

# This script runs LIBERO evaluation tasks in parallel.
# v3: dynamic GPU load management.

run_libero_eval() {
    local task_list_file=$1
    echo "task_file: $task_list_file"

    require_non_empty() {
        local var_name="$1"
        local var_val="${!var_name}"
        if [ -z "$var_val" ]; then
            echo "Error: required variable $var_name is not set"
            exit 1
        fi
    }
    
    # Basic configuration
    ROOT_DIR=${ROOT_DIR:-"$(pwd)"}
    export ROOT_DIR
    # Generate a unique run_id
    RUN_ID=${RUN_ID:-"eval_$(date +%Y%m%d_%H%M%S)"}
    export RUN_ID
    OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/evaluate_results/$RUN_ID"}
    export OUTPUT_DIR  # Use run_id as the output subdirectory
    SESSION_NAME=${SESSION_NAME:-"libero_test_v3_${RUN_ID}_$$"}
    EXP_NAME=${EXP_NAME:-""}
    export EXP_NAME
    MUJOCO_GL=${MUJOCO_GL:-osmesa}
    PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
    export MUJOCO_GL PYOPENGL_PLATFORM
    WORKER_ENV_SOURCE=${WORKER_ENV_SOURCE:-${LIBERO_WORKER_ENV_SOURCE:-"$HOME/.bashrc"}}
    export WORKER_ENV_SOURCE
    WORKER_PYTHONPATH=${WORKER_PYTHONPATH:-${PYTHONPATH:-}}
    export WORKER_PYTHONPATH
    local tmux_bin="${TMUX_BIN:-/usr/bin/tmux}"
    tmux() {
        env -u LD_LIBRARY_PATH "$tmux_bin" "$@"
    }

    echo "EXP_NAME: $EXP_NAME"
    echo "MUJOCO_GL: $MUJOCO_GL"
    echo "PYOPENGL_PLATFORM: $PYOPENGL_PLATFORM"
    echo "WORKER_ENV_SOURCE: $WORKER_ENV_SOURCE"
    echo "WORKER_PYTHONPATH: $WORKER_PYTHONPATH"
    
    # Create the output directory
    mkdir -p "$OUTPUT_DIR"
    echo "Evaluation results will be saved to: $OUTPUT_DIR"

    # Copy task_list_file into OUTPUT_DIR
    cp "$task_list_file" "$OUTPUT_DIR/"
    task_list_file="$OUTPUT_DIR/$(basename $task_list_file)"
    echo "Task list file copied to: $task_list_file"
    TASK_CHUNK_SIZE=${TASK_CHUNK_SIZE:-1}
    if [ "$TASK_CHUNK_SIZE" -le 0 ]; then
        echo "Error: TASK_CHUNK_SIZE must be positive, got $TASK_CHUNK_SIZE"
        exit 1
    fi
    export TASK_CHUNK_SIZE
    echo "TASK_CHUNK_SIZE: $TASK_CHUNK_SIZE"
    
    # GPU and tmux configuration
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        # If CUDA_VISIBLE_DEVICES is not set, require NUM_GPUS explicitly
        require_non_empty "NUM_GPUS"
        AVAILABLE_GPUS=$(seq 0 $((NUM_GPUS-1)) | tr '\n' ',' | sed 's/,$//')
    else
        # If CUDA_VISIBLE_DEVICES is set, parse the visible GPUs
        AVAILABLE_GPUS=$CUDA_VISIBLE_DEVICES
        NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    fi
    export NUM_GPUS
    echo "NUM_GPUS: $NUM_GPUS, AVAILABLE_GPUS: $AVAILABLE_GPUS"

    # Convert AVAILABLE_GPUS to an array
    IFS=',' read -r -a GPU_ARRAY <<< "$AVAILABLE_GPUS"

    require_non_empty "MAX_TASKS_PER_GPU"
    require_non_empty "NUM_TRIALS"
    TMUX_GRID_ROWS=${TMUX_GRID_ROWS:-1}
    TMUX_GRID_COLS=${TMUX_GRID_COLS:-$((MAX_TASKS_PER_GPU + 1))}
    GRID_ROWS=$TMUX_GRID_ROWS
    GRID_COLS=$TMUX_GRID_COLS
    MAX_PANES=$((GRID_ROWS * GRID_COLS - 1))
    if [ "$MAX_PANES" -le 0 ]; then
        echo "Error: invalid tmux grid configuration, TMUX_GRID_ROWS=$TMUX_GRID_ROWS TMUX_GRID_COLS=$TMUX_GRID_COLS"
        exit 1
    fi
    
    # GPU load tracking files
    GPU_LOAD_FILE="$OUTPUT_DIR/gpu_load.txt"
    TASK_GPU_MAP_FILE="$OUTPUT_DIR/task_gpu_map.txt"
    TASK_STATUS_DIR="$OUTPUT_DIR/task_status"
    TASK_LOG_DIR="$OUTPUT_DIR/task_logs"
    FAILED_TASKS_FILE="$OUTPUT_DIR/failed_tasks.txt"

    mkdir -p "$TASK_STATUS_DIR" "$TASK_LOG_DIR"
    : > "$FAILED_TASKS_FILE"
    
    # Initialize GPU load tracking
    init_gpu_load_tracking() {
        # Initialize the current task count of each GPU to 0
        > "$GPU_LOAD_FILE"
        > "$TASK_GPU_MAP_FILE"
        for gpu in "${GPU_ARRAY[@]}"; do
            echo "$gpu:0" >> "$GPU_LOAD_FILE"
        done
        echo "GPU load tracking initialized: $GPU_LOAD_FILE"
    }
    
    # Get the current GPU load
    get_gpu_load() {
        local gpu_id=$1
        local load=$(grep "^$gpu_id:" "$GPU_LOAD_FILE" | cut -d: -f2)
        echo "${load:-0}"
    }
    
    # Update GPU load
    update_gpu_load() {
        local gpu_id=$1
        local new_load=$2
        # Use a temporary file to keep the update atomic
        local temp_file="$GPU_LOAD_FILE.tmp"
        
        # Check whether the file exists first
        if [ -f "$GPU_LOAD_FILE" ]; then
            # Remove the old record and keep records for other GPUs
            grep -v "^${gpu_id}:" "$GPU_LOAD_FILE" > "$temp_file" 2>/dev/null || true
        else
            > "$temp_file"
        fi
        
        # Add the new record
        echo "${gpu_id}:${new_load}" >> "$temp_file"
        mv "$temp_file" "$GPU_LOAD_FILE"
    }
    
    # Increment GPU load
    increment_gpu_load() {
        local gpu_id=$1
        local current_load=$(get_gpu_load $gpu_id)
        local new_load=$((current_load + 1))
        update_gpu_load $gpu_id $new_load
        echo $new_load
    }
    
    # Decrement GPU load
    decrement_gpu_load() {
        local gpu_id=$1
        local current_load=$(get_gpu_load $gpu_id)
        local new_load=$((current_load - 1))
        [ $new_load -lt 0 ] && new_load=0
        update_gpu_load $gpu_id $new_load
        echo $new_load
    }
    
    # Find the least-loaded GPU
    find_least_loaded_gpu() {
        local min_load=999999
        local best_gpu=""
        for gpu in "${GPU_ARRAY[@]}"; do
            local load=$(get_gpu_load $gpu)
            if [ $load -lt $min_load ] && [ $load -lt $MAX_TASKS_PER_GPU ]; then
                min_load=$load
                best_gpu=$gpu
            fi
        done
        echo $best_gpu
    }
    
    # Show GPU load status
    show_gpu_status() {
        echo "=== GPU Load Status ==="
        for gpu in "${GPU_ARRAY[@]}"; do
            local load=$(get_gpu_load $gpu)
            local percentage=$((load * 100 / MAX_TASKS_PER_GPU))
            printf "GPU %s: %d/%d tasks (%d%%)\n" "$gpu" "$load" "$MAX_TASKS_PER_GPU" "$percentage"
        done
        echo "=================="
    }
    
    # Debug helper: show the currently running tasks
    show_debug_info() {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Debug Info ==="
        
        # Show the GPU load file contents
        if [ -f "$GPU_LOAD_FILE" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU load file contents:"
            cat "$GPU_LOAD_FILE" | while IFS=: read gpu load; do
                echo "[$(date '+%Y-%m-%d %H:%M:%S')]   GPU$gpu: $load"
            done
        fi
        
        # Show the chunk mapping file contents
        if [ -f "$TASK_GPU_MAP_FILE" ]; then
            local map_count=$(wc -l < "$TASK_GPU_MAP_FILE" 2>/dev/null || echo 0)
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Number of running chunks: $map_count"
            if [ $map_count -gt 0 ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running chunks:"
                cat "$TASK_GPU_MAP_FILE" | while IFS=: read task_info gpu_id; do
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   $task_info -> GPU$gpu_id"
                done
            fi
        fi
        
        # Show the number of pending chunks
        if [ -f "$PENDING_TASKS_FILE" ]; then
            local pending_count=$(wc -l < "$PENDING_TASKS_FILE" 2>/dev/null || echo 0)
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Number of pending chunks: $pending_count"
        fi

        if [ -f "$FAILED_TASKS_FILE" ]; then
            local failed_count=$(wc -l < "$FAILED_TASKS_FILE" 2>/dev/null || echo 0)
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Number of failed tasks: $failed_count"
        fi
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==================="
    }
    
    # Record the chunk-to-GPU mapping.
    record_task_gpu_mapping() {
        local chunk_id=$1
        local gpu_id=$2
        echo "$chunk_id:$gpu_id" >> "$TASK_GPU_MAP_FILE"
    }
    
    # Get the GPU assigned to a chunk.
    get_task_gpu() {
        local chunk_id=$1
        local mapping=$(grep "^$chunk_id:" "$TASK_GPU_MAP_FILE" | cut -d: -f2)
        echo "${mapping:-}"
    }
    
    # Remove the chunk-to-GPU mapping.
    remove_task_gpu_mapping() {
        local chunk_id=$1
        local temp_file="$TASK_GPU_MAP_FILE.tmp"
        grep -v "^$chunk_id:" "$TASK_GPU_MAP_FILE" > "$temp_file" 2>/dev/null || true
        mv "$temp_file" "$TASK_GPU_MAP_FILE"
    }

    append_unique_pending_chunk() {
        local chunk_record="$1"
        local chunk_id="${chunk_record%%|*}"
        if [ ! -f "$PENDING_TASKS_FILE" ] || ! grep -q "^${chunk_id}|" "$PENDING_TASKS_FILE"; then
            echo "$chunk_record" >> "$PENDING_TASKS_FILE"
        fi
    }

    mark_task_failed() {
        local chunk_id=$1
        local gpu_id=$2
        local return_code=$3
        local log_file=$4
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "$timestamp,$chunk_id,gpu=$gpu_id,rc=$return_code,log=$log_file" >> "$FAILED_TASKS_FILE"
    }
    
    # Checkpoint and config
    CKPT=${CKPT:-""}
    export CKPT
    CONFIG=${CONFIG:-""}
    CONFIG_NAME=${CONFIG_NAME:-"sim_libero"}
    require_non_empty "CKPT"
    require_non_empty "CONFIG"
    # Normalize CONFIG to task/config_name.yaml
    CONFIG="${CONFIG#configs/}" # delete prefix configs/
    CONFIG="${CONFIG#task/}" # delete prefix task/
    CONFIG="${CONFIG%.yaml}" # delete suffix .yaml
    export CONFIG
    export CONFIG_NAME

    echo "CKPT: $CKPT"
    echo "CONFIG_NAME: $CONFIG_NAME"
    echo "CONFIG: $CONFIG"
    echo "ROOT_DIR: $ROOT_DIR"
    echo "NUM_GPUS: $NUM_GPUS"
    echo "MAX_TASKS_PER_GPU: $MAX_TASKS_PER_GPU"
    
    # Initialize GPU load tracking
    init_gpu_load_tracking

    # Always create a fresh scheduler session. If the requested name is already
    # used by the user or another eval, pick a new one instead of killing it.
    BASE_SESSION_NAME="$SESSION_NAME"
    while tmux has-session -t "$SESSION_NAME" 2>/dev/null; do
        SESSION_NAME="${BASE_SESSION_NAME}_$RANDOM"
    done
    export SESSION_NAME
    echo "TMUX session: $SESSION_NAME"

    # Create a new detached session
    tmux new-session -d -s "$SESSION_NAME"
    tmux set-option -t "$SESSION_NAME" renumber-windows off
    tmux set-window-option -t "$SESSION_NAME" pane-base-index 0
    declare -a WINDOW_TARGETS
    WINDOW_TARGETS[0]="$(tmux display-message -p -t "$SESSION_NAME" '#{window_id}')"

    # Create the grid layout
    create_grid_layout() {
        local window_target=$1

        # Get the current number of panes in the window
        local pane_count=$(tmux list-panes -t "$window_target" | wc -l)
        
        # Only create new panes if the current count is below the target count
        for ((i=pane_count; i<GRID_ROWS*GRID_COLS-1; i++)); do
            tmux split-window -d -t "$window_target" >/dev/null
            tmux select-layout -t "$window_target" tiled >/dev/null
        done
    }
    
    # Create the first window layout
    create_grid_layout "${WINDOW_TARGETS[0]}"
    
    # Global pane counter
    NEXT_PANE_INDEX=0
    
    # Helper to ensure a logical window exists.
    ensure_window_exists() {
        local window_id=$1

        if [ -z "${WINDOW_TARGETS[$window_id]:-}" ]; then
            WINDOW_TARGETS[$window_id]="$(tmux new-window -d -P -F '#{window_id}' -t "$SESSION_NAME")"
            create_grid_layout "${WINDOW_TARGETS[$window_id]}"
        fi
    }

    # Return a stable tmux pane id such as %12 for a logical pane index.
    get_pane_target() {
        local next_index=$1
        local window_id=$((next_index / MAX_PANES))
        local pane_id=$((next_index % MAX_PANES))

        ensure_window_exists "$window_id"
        local window_target="${WINDOW_TARGETS[$window_id]}"
        create_grid_layout "$window_target"

        local panes=()
        mapfile -t panes < <(tmux list-panes -t "$window_target" -F '#{pane_id}')
        if [ "${#panes[@]}" -le "$pane_id" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed to resolve pane index $pane_id in $window_target" >&2
            return 1
        fi
        echo "${panes[$pane_id]}"
    }
    
    # Launch a single task.
    # Pane assignment is handled outside this function.
    launch_task_on_pane() {
        local chunk_id=$1
        local chunk_file=$2
        local first_suite=$3
        local first_task_id=$4
        local gpu_id=$5
        local pane_info=$6
        local status_file="$TASK_STATUS_DIR/${chunk_id}.status"
        local log_file="$TASK_LOG_DIR/${chunk_id}_${first_suite}_task${first_task_id}_gpu${gpu_id}.log"
        local worker_env_cmd=""
        local worker_pythonpath_cmd=""
        if [ -n "$WORKER_ENV_SOURCE" ]; then
            worker_env_cmd="source $WORKER_ENV_SOURCE && "
        fi
        if [ -n "$WORKER_PYTHONPATH" ]; then
            worker_pythonpath_cmd="export PYTHONPATH='$WORKER_PYTHONPATH' && "
        fi
        
        rm -f "$status_file"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching chunk: $chunk_id first_task=$first_suite,$first_task_id on GPU$gpu_id pane $pane_info"
        
        # Launch the task in a tmux pane.
        # When the task exits, write a status file so the scheduler can detect failures promptly.
        local tmux_target="$pane_info"
        if ! tmux select-pane -t "$tmux_target"; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed to select tmux pane: $tmux_target"
            return 1
        fi
        if ! tmux send-keys -t "$tmux_target" "clear" C-m; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed to clear tmux pane: $tmux_target"
            return 1
        fi
        if ! tmux send-keys -t "$tmux_target" "${worker_env_cmd}cd $ROOT_DIR && export EXP_NAME=$EXP_NAME MUJOCO_GL=$MUJOCO_GL PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM && \
            ${worker_pythonpath_cmd} \
            STATUS_FILE='$status_file' LOG_FILE='$log_file' && \
            CUDA_VISIBLE_DEVICES=$gpu_id python experiments/libero/eval_libero_single.py \
            --config-name $CONFIG_NAME \
            task=$CONFIG ckpt=$CKPT \
            EVALUATION.task_chunk_file='$chunk_file' EVALUATION.task_suite_name=$first_suite EVALUATION.task_id=$first_task_id gpu_id=$gpu_id \
            EVALUATION.num_trials=$NUM_TRIALS EVALUATION.output_dir=$OUTPUT_DIR $EXTRA_ARGS > \"\$LOG_FILE\" 2>&1; \
            rc=\$?; \
            if [ \$rc -eq 0 ]; then \
                echo \"SUCCESS|$gpu_id|\$rc|\$(date +%s)|\$LOG_FILE\" > \"\$STATUS_FILE\"; \
            else \
                echo \"FAILED|$gpu_id|\$rc|\$(date +%s)|\$LOG_FILE\" > \"\$STATUS_FILE\"; \
            fi" C-m; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed to launch task in tmux pane: $tmux_target"
            return 1
        fi
        return 0
    }

    launch_task() {
        local chunk_id=$1
        local chunk_file=$2
        local first_suite=$3
        local first_task_id=$4
        local gpu_id=$5
        local pane_info=$6

        if ! launch_task_on_pane "$chunk_id" "$chunk_file" "$first_suite" "$first_task_id" "$gpu_id" "$pane_info"; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launch failed before assignment: $chunk_id GPU$gpu_id pane=$pane_info"
            return 1
        fi
        record_task_gpu_mapping "$chunk_id" "$gpu_id"
        local new_load=$(increment_gpu_load "$gpu_id")
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Assigned chunk: $chunk_id -> GPU$gpu_id (load: $new_load/$MAX_TASKS_PER_GPU)"
        return 0
    }
    
    # Check completed tasks and clean up finished entries
    cleanup_completed_tasks() {
        CLEANED_COUNT=0
        NEW_FAILURE_COUNT=0

        if [ ! -f "$TASK_GPU_MAP_FILE" ] || [ ! -s "$TASK_GPU_MAP_FILE" ]; then
            return 0
        fi

        local temp_map="$TASK_GPU_MAP_FILE.cleanup"
        > "$temp_map"

        while IFS=: read -r task_info gpu_id; do
            [ -z "$task_info" ] && continue

            local chunk_id="$task_info"
            local status_file="$TASK_STATUS_DIR/${chunk_id}.status"

            # The chunk process exited: detect it, report failures, and reclaim the mapping.
            if [ -f "$status_file" ]; then
                IFS='|' read -r status status_gpu status_rc status_ts status_log < "$status_file"
                if [ "$status" = "FAILED" ]; then
                    local new_load=$(decrement_gpu_load "$gpu_id")
                    mark_task_failed "$chunk_id" "$gpu_id" "${status_rc:-unknown}" "${status_log:-unknown}"
                    ((NEW_FAILURE_COUNT++))
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Chunk failed: $chunk_id rc=$status_rc GPU$gpu_id (current load: $new_load/$MAX_TASKS_PER_GPU)"
                    rm -f "$status_file"
                    continue
                fi
                if [ "$status" = "SUCCESS" ]; then
                    local new_load=$(decrement_gpu_load "$gpu_id")
                    rm -f "$status_file"
                    ((CLEANED_COUNT++))
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Chunk completed: $chunk_id GPU$gpu_id released (load: $new_load/$MAX_TASKS_PER_GPU)"
                    continue
                fi
            fi

            # Still running: keep the mapping
            echo "$task_info:$gpu_id" >> "$temp_map"
        done < "$TASK_GPU_MAP_FILE"

        mv "$temp_map" "$TASK_GPU_MAP_FILE"
        return 0
    }

    
    # Main loop for dynamic task scheduling
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting dynamic task scheduling..."
    
    local total_tasks=$(wc -l < "$task_list_file")
    local total_chunks=0
    TASK_CHUNK_DIR="$OUTPUT_DIR/task_chunks"
    TASK_CHUNK_LIST_FILE="$OUTPUT_DIR/task_chunks.txt"
    PENDING_TASKS_FILE="$OUTPUT_DIR/pending_chunks.txt"
    mkdir -p "$TASK_CHUNK_DIR"
    > "$TASK_CHUNK_LIST_FILE"

    current_chunk_file=""
    current_chunk_id=""
    current_first_suite=""
    current_first_task_id=""
    current_chunk_count=0
    chunk_idx=0
    while IFS=, read -r suite task_id; do
        [ -z "$suite" ] && continue
        if [ "$current_chunk_count" -eq 0 ]; then
            current_chunk_id=$(printf "chunk_%06d" "$chunk_idx")
            current_chunk_file="$TASK_CHUNK_DIR/${current_chunk_id}.txt"
            current_first_suite="$suite"
            current_first_task_id="$task_id"
            > "$current_chunk_file"
            echo "${current_chunk_id}|${current_chunk_file}|${current_first_suite}|${current_first_task_id}" >> "$TASK_CHUNK_LIST_FILE"
            chunk_idx=$((chunk_idx + 1))
            total_chunks=$((total_chunks + 1))
        fi
        echo "$suite,$task_id" >> "$current_chunk_file"
        current_chunk_count=$((current_chunk_count + 1))
        if [ "$current_chunk_count" -ge "$TASK_CHUNK_SIZE" ]; then
            current_chunk_count=0
        fi
    done < "$task_list_file"
    cp "$TASK_CHUNK_LIST_FILE" "$PENDING_TASKS_FILE"

    local monitoring_interval=${MONITORING_INTERVAL:-10}  # Monitoring interval in seconds
    local last_status_time=0
    local status_interval=${STATUS_INTERVAL:-30}  # Status display interval in seconds
    local max_launch_per_round=${MAX_LAUNCH_PER_ROUND:-$((NUM_GPUS * MAX_TASKS_PER_GPU))}
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Total tasks: $total_tasks"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Total chunks: $total_chunks"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Max tasks per GPU: $MAX_TASKS_PER_GPU"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Available GPUs: ${GPU_ARRAY[*]}"
    
    # Initial launch phase: start initial tasks for each GPU
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting the initial launch phase..."
    local initial_launched=0
    local max_initial_tasks=$((NUM_GPUS * MAX_TASKS_PER_GPU))
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Planning to launch up to $max_initial_tasks initial tasks"
    
    # Simplified version: launch chunks in order and let find_least_loaded_gpu choose the GPU
    local task_array=()

    # Read all chunks into an array first
    while IFS= read -r chunk_record; do
        [ -z "$chunk_record" ] && continue
        task_array+=("$chunk_record")
    done < "$TASK_CHUNK_LIST_FILE"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Loaded ${#task_array[@]} chunks"
    
    # Launch initial tasks
    for task_info in "${task_array[@]}"; do
        [ $initial_launched -ge $max_initial_tasks ] && break
        
        IFS='|' read -r chunk_id chunk_file suite task_id <<< "$task_info"
        [ -z "$chunk_id" ] && continue

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing chunk: $chunk_id first_task=$suite,$task_id"
        
        # Find the least-loaded GPU
        gpu_id=$(find_least_loaded_gpu)
        if [ -z "$gpu_id" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] All GPUs are fully loaded, stopping initial launch"
            break
        fi
        
        # Resolve a stable tmux pane id directly from tmux.
        pane_info="$(get_pane_target "$NEXT_PANE_INDEX")" || {
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Keeping chunk pending after pane resolution failure: $chunk_id"
            continue
        }
        NEXT_PANE_INDEX=$((NEXT_PANE_INDEX + 1))
        
        # Launch the task. Only remove it from pending after tmux accepts the command.
        if launch_task "$chunk_id" "$chunk_file" "$suite" "$task_id" "$gpu_id" "$pane_info"; then
            ((initial_launched++))
            grep -v "^$chunk_id|" "$PENDING_TASKS_FILE" > "$PENDING_TASKS_FILE.tmp" || true
            mv "$PENDING_TASKS_FILE.tmp" "$PENDING_TASKS_FILE"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Keeping chunk pending after launch failure: $chunk_id"
        fi
        
        # Add a small delay to make sure the task starts cleanly
        sleep 0.5
    done
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Initial launch completed, started $initial_launched chunks"
    
    while true; do
        current_time=$(date +%s)

        # Clean up completed/failed tasks and synchronize GPU load
        cleanup_completed_tasks
        cleaned=$CLEANED_COUNT
        new_failures=$NEW_FAILURE_COUNT
        total_failed=$(wc -l < "$FAILED_TASKS_FILE" 2>/dev/null || echo 0)

        if [ "$new_failures" -gt 0 ]; then
            echo "Detected failed subtasks, stopping the scheduler. Failure details: $FAILED_TASKS_FILE"
            cat "$FAILED_TASKS_FILE"
            return 2
        fi

        # Check whether all tasks have completed
        total_completed=$(find "$OUTPUT_DIR" -type f -name "gpu*_task*_results.json" | wc -l)
        if [ "$total_completed" -eq "$total_tasks" ]; then
            echo "All tasks are complete!"
            break
        fi

        # Try to launch new tasks
        launched_this_round=0

        # Read the pending task list.
        # Create a copy to avoid concurrent file access issues.
        temp_pending="$PENDING_TASKS_FILE.processing"
        cp "$PENDING_TASKS_FILE" "$temp_pending" 2>/dev/null || continue

        # Create a new pending task file
        > "$PENDING_TASKS_FILE"

        while IFS='|' read -r chunk_id chunk_file suite task_id; do
            [ -z "$chunk_id" ] && continue

            # Check whether the chunk is already running.
            # The pending file should only keep chunks that are not running.
            running_gpu=$(get_task_gpu "$chunk_id")
            if [ -n "$running_gpu" ]; then
                continue
            fi

            # Find the least-loaded GPU and try to launch
            gpu_id=$(find_least_loaded_gpu)
            if [ -n "$gpu_id" ]; then
                pane_info="$(get_pane_target "$NEXT_PANE_INDEX")" || {
                    append_unique_pending_chunk "$chunk_id|$chunk_file|$suite|$task_id"
                    continue
                }
                NEXT_PANE_INDEX=$((NEXT_PANE_INDEX + 1))

                if launch_task "$chunk_id" "$chunk_file" "$suite" "$task_id" "$gpu_id" "$pane_info"; then
                    ((launched_this_round++))
                else
                    append_unique_pending_chunk "$chunk_id|$chunk_file|$suite|$task_id"
                    continue
                fi

                # Limit the number of launches per round to avoid overloading the system
                if [ $launched_this_round -ge $max_launch_per_round ]; then
                    while IFS= read -r remaining_chunk_record; do
                        [ -n "$remaining_chunk_record" ] && append_unique_pending_chunk "$remaining_chunk_record"
                    done
                    break
                fi
            else
                # GPUs are fully loaded, put the chunk back into the pending queue
                append_unique_pending_chunk "$chunk_id|$chunk_file|$suite|$task_id"
            fi
        done < "$temp_pending"

        # Clean up the temporary file
        rm -f "$temp_pending"

        running_count=$(wc -l < "$TASK_GPU_MAP_FILE" 2>/dev/null || echo 0)
        pending_count=$(wc -l < "$PENDING_TASKS_FILE" 2>/dev/null || echo 0)

        if [ "$running_count" -eq 0 ] && [ "$pending_count" -eq 0 ] && [ "$total_completed" -lt "$total_tasks" ]; then
            echo "Scheduling inconsistency: no running tasks and no pending tasks, but not all tasks are complete."
            echo "Completed: $total_completed/$total_tasks, failed: $total_failed"
            [ -s "$FAILED_TASKS_FILE" ] && cat "$FAILED_TASKS_FILE"
            return 2
        fi
        
        # Periodically display status
        if [ $((current_time - last_status_time)) -ge $status_interval ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Scheduling Status $(date '+%H:%M:%S') ==="
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Total tasks: $total_tasks"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: $total_completed"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Remaining: $((total_tasks - total_completed))"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running: $running_count"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pending: $pending_count"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed: $total_failed"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launched this round: $launched_this_round"
            if [ "$cleaned" -gt 0 ] 2>/dev/null; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleaned this round: $cleaned"
            fi
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] === GPU Load Status ==="
            for gpu in "${GPU_ARRAY[@]}"; do
                load=$(get_gpu_load $gpu)
                percentage=$((load * 100 / MAX_TASKS_PER_GPU))
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu: $load/$MAX_TASKS_PER_GPU tasks ($percentage%)"
            done
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] =================="
            
            # Add detailed debug info to the status report
            show_debug_info
            echo ""
            last_status_time=$current_time
        fi
        
        # Wait before the next scheduling round
        sleep $monitoring_interval
    done
    
    # Clean up temporary files
    rm -f "$PENDING_TASKS_FILE" "$PENDING_TASKS_FILE.processing"

    # Check the final result
    echo "All tasks completed successfully!"
    # Run the result summarization script
    echo "Generating evaluation report..."
    python experiments/libero/summarize_results.py --output_dir="$OUTPUT_DIR"
}


# Entrypoint
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Check whether a task file argument is provided
    if [ $# -lt 1 ]; then
        echo "Error: task file path is required"
        echo "Usage: $0 <task_file>"
        exit 1
    fi
    test_file="$1"
    run_libero_eval "$test_file"
    exit $?
fi
