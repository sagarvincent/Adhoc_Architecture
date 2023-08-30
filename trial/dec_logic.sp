#const n = 8.
t = 0..n.
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

same_loc_after_act(t, agent).

ts(t).

has_ball(t,agent).
ball_position(t,xpos,ypos).


% -------------------- causal laws -------------------- %

% An agent cannot be in two positions at the same time.
:- move(agent, X1, Y1), move(agent, X2, Y2), ts(t1), ts(t2), t1 = t2, (X1, Y1) != (X2, Y2).

% Two agents cannot be in the same position at the same time.
:- move(agent1, X, Y), move(agent2, X, Y), ts(t), agent1 != agent2, t = n.

% Passing will change the possession of the ball.
has_ball(t, Teammate) :- pass(agent, Teammate), has_ball(t-1, agent).
:- pass(agent, Teammate), has_ball(t-1, Teammate), not has_ball(t, Teammate).

% Two agents cannot have the ball at the same time.
:- has_ball(t, agent1), has_ball(t, agent2), agent1 != agent2.

% The location of an agent and the ball is the same after a move.
same_location_after_move(t, agent) :- move(agent, X, Y), ball_position(t-1, X, Y).
:- move(agent, _, _), ball_position(t, X, Y), not same_loc_after_act(t, agent).

%   



% -------------------- Results/outputs -------------------- %












