classdef Neuron < handle
% This class definition defines a generic neuron
% Synaptic reciprocity guaranteed with only presynaptic methods accessible
% Duplicate synapse not possible by design

    properties
        out = 0;% The 'excitation' state of a neuron to be output to the next level
        presynN = [];% array of presynaptic neurons
        presynW = [];% array of presynaptic weights
        location = [0 0 0 0] % the location of the neuron in the neural network. 
        % [a b c d] which are layer# map# row and colomn. '0' means not decided
        
        
        % In the network, a neuron doesn't need to know its post neurons.
%         postsynN = [];% array of postsynaptic neurons
%         postsynW = [];% array of postsynaptic weights
    end
    
    properties(GetAccess = 'protected', SetAccess = 'public')
        in = 0;% The value to be used in the activation function, usually a weighted sum of the outputs of a few other neurons from the previous level
%         offset = 0; % Offset value to be added in the activation function
        act = @(x) 1.7159*tanh(2/3*x); % function handle for activation function
        % Default function used by Lenet 5 
    end
    
    
    methods
        % constructor
        function obj = Neuron(size,numlayer,nummap)
            % size is the size of a feature map. If missing, default is
            % 1. The length of size is gonna be 2 [a b], which means a map
            narginchk(0,3);
            if nargin == 0
            elseif nargin == 1 || nargin == 3
                if nargin == 1
                    numlayer = 0;
                    nummap = 0;
                end
                for i = 1:size(1)
                    for j = 1:size(2)
                        obj(i,j) = Neuron;
                        obj(i,j).location = [numlayer nummap i j];
                    end
                end
            else
                error('invalid number of inpus');
            end
        end
        % set the neuron state
        function obj = setNode(obj, ex)
            obj.out = ex;
        end
        % set the neuron activation function
        function obj = setAct(obj, act)
            obj.act = act;
        end
        % set the neuron offset
        function obj = setOff(obj, off)
            obj.offset = off;
        end
        % a presynaptic neuron is added with default weight 1
        function obj = addPre(obj, pre, w)
            if pre == obj
                error('Neuron:addPreobj', 'Trying to connect a neuron to itobj')
            end
            if nargin < 3
                w = 1;
            end
            preInd = find(obj.presynN==pre);
            if isempty(preInd)
                obj.presynN = [obj.presynN pre];
                obj.presynW = [obj.presynW w];
            else
                warning('Neuron:addPreExist', 'The neuron to be added is already presynaptic')
                obj.presynW(preInd) = w;
            end
            obj.checkPresyn();
            pre.addPost(obj, w); % reciprocity is guaranteed
        end
        % a presynaptic neuron is disconnected
        function obj = removePre(obj, pre)
            rmInd = find(obj.presynN==pre);
            if isempty(rmInd)
                warning('Neuron:removePreNotExist', 'The neuron to be disconnected is not presynaptic')
                return
            end
            obj.presynN(rmInd) = [];
            obj.presynW(rmInd) = [];
            obj.checkPresyn();
%             pre.removePost(obj); % reciprocity is guaranteed
        end
        % update the neuron
        function obj = update(obj)
            obj.calculateIn();
            obj.activation(obj.in);
        end
        % destructor removes all synapses
        function delete(obj)
            while ~isempty(obj.presynN)
                obj.removePre(obj.presynN(1)); % only need to use removePre
            end
%             while ~isempty(obj.postsynN)
%                 obj.postsynN(1).removePre(obj); % only need to use removePre
%             end
        end
    end
    
    methods(Access = 'protected')
        % a postsynaptic neuron is added with default weight 1
%         function obj = addPost(obj, post, w)
%             if nargin < 3
%                 w = 1;
%             end
%             postInd = find(obj.postsynN==post);
%             if isempty(postInd)
%                 obj.postsynN = [obj.postsynN post];
%                 obj.postsynW = [obj.postsynW w];
%             else
%                 warning('Neuron:addPostExist', 'The neuron to be added is already postsynaptic')
%                 obj.postsynW(postInd) = w;
%             end
%             obj.checkPostsyn();
%         end
        % a postsynaptic neuron is disconnected
%         function obj = removePost(obj, post)
%             rmInd = find(obj.postsynN==post);
%             if isempty(rmInd)
%                 warning('Neuron:removePostNotExist', 'The neuron to be disconnected is not postsynaptic')
%             end
%             obj.postsynN(rmInd) = [];
%             obj.postsynW(rmInd) = [];
%             obj.checkPostsyn();
%         end
        % calculate the input using in-list
        function obj = calculateIn(obj)
            inL = length(obj.presynN);
            obj.checkPresyn();
            obj.in = 0;
            for i=1:inL
                obj.in = obj.in + obj.presynW(i) * obj.presynN(i).out;
            end
        end
        % activation function
        function obj = activation(obj, input)
            obj.out = obj.act(input) + obj.offset;
        end
        % check if presynaptic lists dimensions agree
        function checkPresyn(obj)
            if length(obj.presynN) ~= length(obj.presynW)
                error('Neuron:presynNWMatch', 'Presynaptic lists do not agree')
            end
        end
        % check if postsynaptic lists dimensions agree
%         function checkPostsyn(obj)
%             if length(obj.postsynN) ~= length(obj.postsynW)
%                 error('Neuron:postsynNWMatch', 'Postsynaptic lists do not agree')
%             end
%         end
    end
end