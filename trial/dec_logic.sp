#const n = 8.
steps(1..n).
% -------------------- inputs -------------------- %
xpos = 1..5.
ypos = 1..5.
maxopenangle = 1..5.
oppdist = 1..5.
teamopenangle = 1..5.
maxpassangle = 1..5.

% -------------------- predicates -------------------- %
agent(adhoc).
teammates= {t1;t2;t3;t4}.
opponents(opp1;opp2;opp3;opp4;opp5).

pass(agent,teammate).
shoot(agent).
dribble(agent).
dash(agent).
move(agent,xpos,ypos).

has_ball(agent).

% -------------------- causal laws -------------------- %

% agent cant be in two positions

% two agents cant be in the same positions

% passing will change the possesion of ball

% two agents cant have the ball at the same time

% location of agent and the ball is the same 

%  














