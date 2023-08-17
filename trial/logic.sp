

% --------------------- facts and constants --------------------- %
#xy_vals = 0..19.
#threshold = 10
#agent = {adhoc}.
#teammate = {offense1, offense2, offense3}.
#defense = {defense1,defense2, defense3,defense4}.
#other_agents = #teammate + #defense.
#offense = #agent + #teammate.
#angle_val = 0..180.
#step = 0..n.
#nearoppdist = {oppdist}

#agent_actions = shoot(#agent) + pass(#agent, #teammate) + dribble(#agent, #x_value, #y_value) + move(#agent, #x_value, #y_value).


% --------------------- predicates and constraints for tactical decision -------------------- %
action(attack).
action(hold).

nearopp() :- oppdist <= threshold
hasfree() :- not nearopp(), validangleopp(), .

teammateavail() :- teammatenear(), angletoteam().

dribblechance() :- clearspace(), teammateno().

% hold the ball if certain conditions are met
hold() :- not hasfree(), not teammateavil(), not dribblechance().

% attack if certain conditions are met
attack() :- hasfree(), teammateavail(), dribblechance().

hide.
