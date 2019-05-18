class Node:
    def __init__(self, g, pos, goal, b, parent):
        direction = [(1,0), (-1,0), (0,1), (0,-1)]
        self.g = g
        self.pos = pos
        self.goal = goal
        self.f = g + abs(pos[0]-goal[0]) + abs(pos[1]-goal[1])
        self.b = list()
        for i, x in enumerate(b):
            if (x == '1'): self.b.append(direction[i])
        self.parent = parent

def solve_maze(maze, r0, c0, r, c):
    start = Node(0, (r0, c0), (r, c), maze[r0][c0], None)
    open_list = [start]
    closed_list = list()
    curr_best = [start]
    while (True):
        open_list.sort(key=lambda x: x.f)
        curr_best = [open_list[0]]
        del open_list[0]
        if (len(open_list) > 1):
            for index, node in enumerate(open_list):
                if (node.f == curr_best[0].f):
                    curr_best.append(node)
                    del open_list[index]
                else: break
        for node in curr_best:
            closed_list.append(node)
            for i, j in node.b:
                i1, j1 = node.pos[0]+i, node.pos[1]+j
                if (i1 >= 0 and i1 < len(maze) and j1 >=0 and j1 <= len(maze[0]) and
                    all((i1, j1) != x.pos for x in closed_list) and
                    all((i1, j1) != x.pos for x in open_list)):
                    open_list.append(Node(node.g+1, (i1, j1), (r, c), maze[i1][j1], node))
                    if((i1, j1) == (r, c)):
                        ll = node
                        path = [ll.pos]
                        while(ll.parent != None):
                            ll = ll.parent
                            path.append(ll.pos)
                        print(list(reversed(path)))
                        return

def main():
    maze = [['1010','0011','0101'],
            ['1100','0010','1001'],
            ['0110','0011','1101'],
            ['1011','0001','1100'],
            ['0110','0011','0101']]

solve_maze(maze, 0, 0, 3, 0)

if __name__ == '__main__':
    main()
