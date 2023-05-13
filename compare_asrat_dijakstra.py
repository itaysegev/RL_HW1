
import datetime
import a_star
import dijkstra

from puzzle import Puzzle
from state import State
import planning_utils

def create_hard_puzzle():
    # Create a hard puzzle that requires at least 25 actions to solve
    initial_state = State()
    actions = [
        'r', 'r', 'd', 'l', 'u', 'l', 'd', 'd', 'r', 'r', 'u', 'l', 'd', 'r', 'u', 'u', 'l', 'd', 'l', 'd', 'r', 'r',
        'u', 'l', 'u','l','d','d','r','u','u','r','d'
    ]
    goal_state = initial_state
    for a in actions:
        valid_actions = goal_state.get_actions()
        assert a in valid_actions, f"Invalid action: {a}"
        goal_state = goal_state.apply_action(a)
    return Puzzle(initial_state, goal_state)


def compare_algorithms(puzzle):
    # Run Dijkstra's algorithm
    prev_dijkstra = dijkstra.dijkstra(puzzle)

    # Run A* algorithm
    prev_a_star = a_star.a_star(puzzle)

    # Extract the state-action sequences
    plan_dijkstra = planning_utils.traverse(puzzle.goal_state, prev_dijkstra)
    plan_a_star = planning_utils.traverse(puzzle.goal_state, prev_a_star)

    # Compare the plans
    print('Dijkstra plan:')
    planning_utils.print_plan(plan_dijkstra)
    print('A* plan:')
    planning_utils.print_plan(plan_a_star)

    # Compare the lengths of the plans
    print('Dijkstra plan length:', len(plan_dijkstra) - 1)
    print('A* plan length:', len(plan_a_star) - 1)

    # Measure the running times
    start_time = datetime.datetime.now()
    dijkstra.dijkstra(puzzle)
    dijkstra_time = datetime.datetime.now() - start_time

    start_time = datetime.datetime.now()
    a_star.a_star(puzzle)
    a_star_time = datetime.datetime.now() - start_time

    print('Dijkstra running time:', dijkstra_time)
    print('A* running time:', a_star_time)

    # Measure the number of states visited
    dijkstra_states = len(prev_dijkstra)
    a_star_states = len(prev_a_star)

    print('Dijkstra states visited:', dijkstra_states)
    print('A* states visited:', a_star_states)


if __name__ == '__main__':
    puzzle = create_hard_puzzle()
    compare_algorithms(puzzle)
