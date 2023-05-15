from puzzle import *
from planning_utils import *
import heapq
import datetime


def a_star(puzzle):
    '''
    Apply A* to a given puzzle
    :param puzzle: The puzzle to solve
    :return: A dictionary mapping state (as strings) to the action that should be taken (also a string)
    '''

    initial = puzzle.start_state.to_string()
    goal = puzzle.goal_state.to_string()

    # The fringe is the queue to pop items from
    fringe = [(0, puzzle.start_state)]
    # Concluded contains states that were already resolved
    concluded = set()
    # A mapping from state (as a string) to the currently minimal distance (int).
    distances = {initial: 0}
    # The return value of the algorithm, a mapping from a state (as a string) to the state leading to it (NOT as string)
    # that achieves the minimal distance to the starting state of the puzzle.
    prev = {initial: None}

    while len(fringe) > 0:
        _, current = heapq.heappop(fringe)
        current_str = current.to_string()

        # If we have already visited this node, skip
        if current_str in concluded:
            continue

        # Mark current node as visited
        concluded.add(current_str)

        # If we have reached the goal, we are done
        if current_str == goal:
            break

        for action in current.get_actions():
            # Generate new state based on action
            next_state = current.apply_action(action)
            next_state_str = next_state.to_string()

            # Calculate tentative distance through current node
            tentative_distance = distances[current_str] + 1

            if next_state_str not in distances or tentative_distance < distances[next_state_str]:
                # This path is the best until now. Record it!
                distances[next_state_str] = tentative_distance
                priority = tentative_distance + next_state.get_manhattan_distance(puzzle.goal_state)
                heapq.heappush(fringe, (priority, next_state))
                prev[next_state_str] = current_str
    print("Number visited states:{}".format(len(concluded)))
    return prev


def solve(puzzle):
    # Compute mapping to previous using A*
    prev_mapping = a_star(puzzle)
    # Extract the state-action sequence
    plan = traverse(puzzle.goal_state, prev_mapping)
    print_plan(plan)
    return plan


if __name__ == '__main__':
    # Create some start and goal states. The number of actions between them is 25, although a shorter plan of
    # length 19 exists (make sure your plan is of the same length)
    initial_state = State()
    actions = [
        'r', 'r', 'd', 'l', 'u', 'l', 'd', 'd', 'r', 'r', 'u', 'l', 'd', 'r', 'u', 'u', 'l', 'd', 'l', 'd', 'r', 'r',
        'u', 'l', 'u'
    ]
    goal_state = initial_state
    for a in actions:
        goal_state = goal_state.apply_action(a)
    puzzle = Puzzle(initial_state, goal_state)
    print('Original number of actions: {}'.format(len(actions)))
    solution_start_time = datetime.datetime.now()
    solve(puzzle)
    print('Time to solve: {}'.format(datetime.datetime.now() - solution_start_time))



