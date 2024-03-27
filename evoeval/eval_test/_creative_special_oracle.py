# oracle for EvoEval/51 in creative
def _check_maze(maze, start, end, solution_path, gt_path):
    if not gt_path:
        assert solution_path == []
    else:
        # check the path according to solution reaches from start to end
        move_to_direction = {
            "right": (0, 1),
            "left": (0, -1),
            "up": (-1, 0),
            "down": (1, 0),
        }
        current_position = start
        for move in solution_path:
            current_position = (
                current_position[0] + move_to_direction[move][0],
                current_position[1] + move_to_direction[move][1],
            )
            assert maze[current_position[0]][current_position[1]] != 1

        assert current_position == end


# oracle for EvoEval/55 in creative
def _check_path(maze, start, end, solution_path, gt_path):
    if not gt_path:
        assert solution_path == []
    else:
        # check the path according to solution reaches from start to end
        assert solution_path[0] == start
        assert solution_path[-1] == end
        assert maze[start[0]][start[0]] != 0
        for i in range(1, len(solution_path)):
            prev_x, prev_y = solution_path[i - 1]
            curr_x, curr_y = solution_path[i]
            assert maze[curr_x][curr_y] != 0  # not a wall
            assert abs(curr_x - prev_x) + abs(curr_y - prev_y) == 1  # adjacent


# oracle for EvoEval/110 in creative
def _check_product(arr, target, solution, gt):
    if gt == "No magic today":
        assert gt == solution
    else:
        assert isinstance(solution, tuple)
        i, j = solution
        assert 0 <= i < j < len(arr)  # don't allow negative indexing
        assert arr[i] * arr[j] == target
