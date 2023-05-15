from state import State
def traverse(goal_state, prev):
    '''
    Extract a plan using the result of Dijkstra's algorithm
    :param goal_state: The end state
    :param prev: Result of Dijkstra's algorithm
    :return: A list of (state, actions) such that the first element is (start_state, a_0), and the last is
    (goal_state, None)
    '''
    result = [(goal_state, None)]
    current_state = goal_state.to_string()

    # Backtrack from the goal state to the start state
    while current_state is not None:
        prev_state_str = prev.get(current_state, None)
        prev_state = State(prev_state_str)
        for action in prev_state.get_actions():
            if prev_state.apply_action(action).to_string() == current_state:
                result.append((prev_state, action))  # Found the action that leads to the current state

        current_state = prev_state_str

    # Reverse the result to get the correct order
    result.reverse()

    return result



def print_plan(plan):
    print('plan length {}'.format(len(plan) - 1))
    for current_state, action in plan:
        print(current_state.to_string())
        if action is not None:
            print('apply action {}'.format(action))
