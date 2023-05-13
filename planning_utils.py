from state import State
def traverse(goal_state, prev):
    '''
    Extract a plan using the result of Dijkstra's algorithm
    :param goal_state: The end state
    :param prev: Result of Dijkstra's algorithm
    :return: A list of (state, actions) such that the first element is (start_state, a_0), and the last is
    (goal_state, None)
    '''
    result = []
    current_state = goal_state.to_string()

    # Backtrack from the goal state to the start state
    while current_state is not None:
        state = State(current_state)
        action = prev[current_state]
        result.append((state, action))
        current_state = prev[current_state]

    # Reverse the result to get the correct order
    result.reverse()

    return result



def print_plan(plan):
    print('Plan length: {}'.format(len(plan) - 1))
    for current_state, action in plan:
        print(current_state.to_string())
        if action is not None:
            print('Apply action: {}'.format(action))
