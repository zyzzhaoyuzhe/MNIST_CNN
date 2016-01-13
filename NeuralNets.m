classdef NeuralNets < handle
    % I want this class to be as general as possible. The network structure
    % can be stored in the connections in the neurons.
    % So the class contains many neuron handles.
    % The method NetBuild will build the network structure.
    % The method BackProp will perform backpropagation
    % The method Train will perform stochastic Gradient Descend.
    
    properties
        structure % Net structure. d dimensional cell array. each cell contains
        % c-D cell arrays for c maps in that array. each map cell contains
        % a matrix [a,b]
        % d is the number of layer.  c is the number of maps in that
        % layer. [a b] is the size of one sheet.
        type % consist of the type of the map
        config % configuration of each map (how it is connected to its previous)
        trainset
        label % Label for training data
        testset
    end
    
    properties(SetAccess = 'protected') % Integrate Neuron Class into this class for performance.
        presynN % presynaptic neuron
        w_Grad % Gradient of the weights (used for gradient descend)
        w_Hess % Hessian of the weights (used for update learning step)
        weights % weights
        windex % indeces of non zeros elements in weights matrices. 
        backweights % backweights for RBF layer
        wshare % wweights share
        in % input of each neuron
        out % ouput of each neuron, which is act(in);
        diffout % which is diffact(in);
        diffout2 % second derivative of the activation function
        step % step size for each weight (parameter)
        act % activation function of each neuron
        diffact % first derivative of the activation function
        cache = [] % Cache for the program. Cache must be assigned empty if not used any longer
        nconnection % number of connections in each layer
        nneuron % number of neurons in each layer
        ntrain % number of trainable parameters
    end
    
    properties(Access = 'public')
        trainfile = '../DATA/train.csv';
        testfile = '../DATA/test.csv';
        jconst = 1 % is the constant in the loss function
        eta = 0.01 % parameter in step calculate (step size)
        mu = 1 % 
    end
    
    methods % Neural Nets Methods
        %% constructor
        function obj = NeuralNets()
            
        end
        %% Data reader
        function obj = DataRead(obj)
            % Data.m is the preprocessed version.
            if exist('Data.mat','file')
                load('Data.mat');
                obj.trainset = trainset;
                obj.label = label;
                obj.testset = testset;
                clear trainset label testset;
            else
                obj.trainset = csvread(obj.trainfile,1,1);
                nrow = size(obj.trainset,1);
                obj.label = csvread(obj.trainfile,1,0,[1,0,nrow,0]);
                obj.testset = csvread(obj.testfile,1,0);
            end
        end
        %% Weights Read
        function obj = WeightAssign(obj,w)
            obj.weights = w;
        end
        %% Net structure creator
        function obj = NetStruct(obj)
            
            
            
        end
        %% Network Builder (no Active function)
        function obj = NetBuild(obj,structure,type,config)
            % We assume the first layer is the input layer.
            
            %             % check the input
            %             if length(structure)~=2
            %                 error('Invalid structure');
            %             end
            %             if length(size(type))~=2
            %                 error('Invalid type');
            %             end
            dim = length(structure);
            if dim ~= length(type)
                error('The dimensions of structure and type do not match');
            end
            obj.structure = structure;
            obj.type = type;
            obj.config = config;
            
            % The first layer: no pre-synaptic connections
            obj.IniLayer(1);
            
            % construct other layers
            for i = 2:length(structure)
                if strcmp(type{i},'C')
                    obj.AddConv(i);
                elseif strcmp(type{i},'S')
                    obj.AddPull(i);
                elseif strcmp(type{i},'F')
                    if isequal(obj.act{i},@CalRBF)
                        obj.AddFull(i,1);
                    else
                        obj.AddFull(i);
                    end
                elseif strcmp(type{i},'L')
                    obj.AddFull(i,1);
                end
            end
            
            % weight sharing initiate
            obj.ShareWeight_Ini;
            % backweights
            obj.CalBackweights;
            % convert wshare
            obj.Convertwshare;
        end
        %% Load weights
        function obj = Loadweights(obj,filename)
            % filename contains the preset weights
            load(filename);
            if exist('weights','var')
                obj.weights = weights;
            else
                warning('check the name of the variable in the file');
            end
        end
        %% Feed Forward
        function obj = FeedFor(obj,data,label,secdiff)
            % data has the same size as the first layer
            % data is gonna be a single data point
            % secdiff is optional and is a boolean variable
            
            % assign data as the output of the input layer
            % check the size
            if nargin == 3
                secdiff = 0;
            end
            datasize = size(data);
            labelleng = length(label);
            if size(data,3) ~= labelleng
                error('the length of label does not equal the number of data');
            end
            mapsize = obj.structure{1}{1};
            if ~isequal(length(datasize(:)),length(mapsize(:)))
                error('The size of the data is invalid');
            end
            % assign the output of the first layer
            obj.out{1}(1:end-1) = data(:);
            obj.out{1}(end) = 1;
            % feed forward the info to the maps one by one
            nlayer = length(obj.structure);
            for d = 2:nlayer
                if isequal(obj.act{d},@CalRBF) % RBF layer
                    % calculated out and diffout for RBF layer
                    obj.CalRBF(d);
                elseif strcmp(obj.type{d},'L') % loss layer
                    obj.CalLoss(d,label);
                    if secdiff
                        obj.D2Loss(d,label); % diffout2 for Loss layer
                        obj.D2RBF(d-1); % diffout2 for RBF layer
                    end
                else
                    % calculate the in
                    obj.CalLayerIn(d);
                    % calculate the out and diffout
                    obj.CalLayerout(d);
                end
            end
        end
        %% Backpropgation: which is used to update gradient for weights
        function obj = BackProp(obj,secdiff)
            % secdiff is optional and is a boolean variable; if secdiff is
            % ture, update diffout2 as well
            if nargin == 1
                secdiff = 0;
            end
            nlayer = length(obj.structure);
            for d = nlayer:-1:2
                if ~isequal(obj.act{d},@CalRBF) && ~strcmp(obj.type{d},'L') % Normal Layer
                    obj.GradLayer(d);
                    if secdiff
                        obj.HessLayer(d);
                        obj.BackLayer2(d);
                    end
                    obj.BackLayer(d);
                else
                    if isequal(obj.act{d},@CalRBF) % RBF Layer
                        obj.GradRBF(d);
                        if secdiff
                            obj.HessRBF(d);
                            obj.BackRBF2(d);
                        end
                    end
                    obj.BackSpec(d);
                end
            end
        end
        %% Perform gradient ascend to train the netwok
        function obj = Train(obj,rep,numsub,filename)
            % N is the sample size for one gradient descend (perform
            % stochastic gradient descend)
            
            nsample = size(obj.trainset,1);
            nlayer = length(obj.structure);

            %%%%%%%%%%%%%
            rec = [];
            %%%%%%%%%%%%5
            Tset = obj.trainset;
            L = obj.label;
            
            errorold = 1;
            for i = 1:rep
                tic
                obj.Iniw_Grad;
                idx = randsample(nsample,nsample);
                % update step
                obj.CalStep(obj.trainset(idx(1:500),1),obj.label(idx(1:500)));
                
                count = 0;
                while count<=nsample-numsub
                    left = count+1;
                    count = count+numsub;
                    right = count;
                    
                    % initiate w_Grad
                    for j = 1:right-left+1
                        n = idx(left+j-1);
                        data = Tset(n,:)';
                        obj.FeedFor(data,L(n));
                        obj.BackProp;
                        for kk = 1:nlayer
                            w_GradT{j,kk} = obj.w_Grad{kk};
                        end
                    end
                    % Integrate w_GradT
                    for d = 1:nlayer
                        w_Grad{d} = w_GradT{1,d};
                    end
                    for j = 2:right-left+1
                        for d = 1:nlayer
                            w_Grad{d} = w_Grad{d}+w_GradT{j,d};
                        end
                    end
                    w_Grad = obj.ShareWeight_Add(obj.w_Grad);
%                     w_Grad = obj.ShapeWGrad(w_Grad);
                    % Perform gradient descend
                    obj.UpdateWeights(w_Grad);
                end
                % calculate loss
                [loss,error] = obj.NetPerform(1);
                % save the net if the performance is acceptable
                if error<0.05 && error<errorold
                    save(filename,'obj');
                    errorold = error;
                end
                
                toc
            end
        end
        %% Prediction
        function output = Predict(obj,data)
            % number of rows is the sample #
            nsample = size(data,1);
            output = zeros(nsample,1);
            parfor i = 1:nsample
                obj.FeedFor(data(i,:)',0);
                temp = obj.out{8};
                temp = temp(1:10);
                [~,idx] = min(temp);
                output(i) = idx-1;
            end
        end
    end
    
    methods(Access = 'public') % FeedForward related methods
        %% Initiate a map
        function obj = IniLayer(obj,whichlayer,type)
            % initiate a map without connection
            % mapsize is the size of the map
            % type==0 means regular sigmoid activation function
            % type==1 means other activation functions
            % always has a bias neuron (always on neuron)
            narginchk(2,3);
            if nargin == 2;
                type = 0;
            end
            mapsize = obj.structure{whichlayer}{1};
            nmap = length(obj.structure{whichlayer});
            nneuron = nmap*mapsize(1)*mapsize(2)+1;
            obj.in{whichlayer} = zeros(nneuron,1);
            obj.out{whichlayer} = zeros(nneuron,1);
            if type == 0
                obj.diffout{whichlayer} = zeros(nneuron,1);
                obj.diffout2{whichlayer} = zeros(nneuron,1);
            elseif type == 1
                obj.diffout{whichlayer} = [];
                obj.diffout2{whichlayer} = [];
            end
            % creat cell array for presynN and presynW
            nrow = mapsize(1);
            ncol = mapsize(2);
            for whichmap = 1:length(obj.structure{whichlayer})
                obj.presynN{whichlayer}{whichmap} = cell(nrow,ncol);
            end
            obj.weights{whichlayer} = [];
            obj.w_Grad{whichlayer} = [];
            obj.w_Hess{whichlayer} = [];
            obj.step{whichlayer} = [];
            obj.wshare{whichlayer} = [];
        end
        %% Set activation func and first derivative to ONE layer
        function obj = SetAct(obj,whichlayer,act,diffact)
            % act is a function handle
            % diffact is a function handly of the first derivative
            whichlayer = whichlayer(:);
            whichlayer = whichlayer';
            for d = whichlayer
                obj.act{d} = act;
                obj.diffact{d} = diffact;
            end
        end
        %% Initiate connection (add the connections in loc to this neuron)
        function obj = Add2ConnList(obj,whichlayer,whichmap,a,b,loc)
            % initiate the connection
            % a is the row number of the neuron
            % b  is the column number of the neuron
            % loc is the locations of the presyn neurons
            
            % the location of presyn neurons
            obj.presynN{whichlayer}{whichmap}{a,b} = ...
                [obj.presynN{whichlayer}{whichmap}{a,b}; loc];
        end
        %% Add a convolutional Layer
        function obj = AddConv(obj,whichlayer)
            % whichlayer is a number indicating the layer
            % whichmap indicates which map you want to generate
            % config contains all the parameter needed
            % the neurons in this layer has a connection to the biased
            % neuron, which is the last one in the previous layer
            
            % initiate a new neuron map (without any connection)
            obj.IniLayer(whichlayer);
            
            %
            Slocx = [];
            Slocy = [];
            Sweight = [];
            SWL = []; % weightlabel for sparse weights matrix
            premapsize = obj.structure{whichlayer-1}{1};
            prenmap = length(obj.structure{whichlayer-1});
            prenrow = premapsize(1);
            prencol = premapsize(2);
            Sncol = prenrow*prencol*prenmap+1; % weight matrix ncol +1 because of the bias neuron
            
            mapsize = obj.structure{whichlayer}{1};
            nrow = mapsize(1);
            ncol = mapsize(2);
            nmap = length(obj.structure{whichlayer});
            Snrow = nrow*ncol*nmap+1; % weight matrix nrow
            % always on neuron location
            BiasN = size(obj.in{whichlayer-1},1); % the last one
            labelstart = 0;
            for whichmap = 1:nmap
                
                neighborsize = obj.config{whichlayer}{whichmap}.neighbor;
                neighrow = neighborsize(1);
                neighcol = neighborsize(2);
                
                premap = obj.config{whichlayer}{whichmap}.premaps; %which previous maps.
                % make connection
                for a = 1:nrow
                    % x, y is the position of the center of its receptive
                    % field in the previous map
                    top = a;
                    down = a+neighrow-1;
                    for b = 1:ncol
                        % normal connections
                        left = b;
                        right = b+neighcol-1;
                        
                        % x,y coordinates of the connections
                        [x,y] = meshgrid(top:down,left:right); % location of pre neurons
                        x = x';
                        y = y';
                        x = x(:);
                        y = y(:);

                        xnum = a+nrow*(b-1)+nrow*ncol*(whichmap-1);
                        xtemp = xnum*ones(size(x));
                        
                        L = -2.4/(length(xtemp)*length(premap)+1);
                        U = 2.4/(length(xtemp)*length(premap)+1);
                        
                        
                        for d = premap
                            z = ones(size(x))*d;
                            loc = [x,y,z];
                            
                            % initiate the connection
                            obj.Add2ConnList(whichlayer,whichmap,a,b,loc);
                            
                            % update ytemp
                            ytemp = x+prenrow*(y-1)+prenrow*prencol*(z-1);
                            
                            %
                            Slocx = [Slocx;xtemp];
                            Slocy = [Slocy;ytemp];
                            
                            % initial weights for the connections
                            wtemp = obj.WRand(size(ytemp),L,U);
                            Sweight = [Sweight;wtemp];
                        end
                        % add the connection to the bias neuron
                        Slocx = [Slocx;xnum];
                        Slocy = [Slocy;BiasN];
                        Sweight = [Sweight;obj.WRand([1,1],L,U)];
                        
                        % label weights for weights sharing
                        len = length(x)*length(premap)+1;
                        weightlabel = (labelstart+1):(labelstart+len);
                        weightlabel = weightlabel(:);
                        
                        SWL = [SWL;weightlabel];
                    end
                end
                labelstart = max(SWL);
            end
            
            obj.weights{whichlayer} = sparse(Slocx,Slocy,Sweight,Snrow,Sncol);
            obj.windex{whichlayer} = [Slocx,Slocy];
%             obj.w_Grad{whichlayer} = spalloc(Snrow,Sncol,length(Slocx));
            
            obj.wshare{whichlayer} = SWL;
            
        end
        %% Add a pulling Layer
        function obj = AddPull(obj,whichlayer)
            % whichlayer is a number indicating the layer
            % whichmap indicates which map you want to generate
            % config contains all the parameter needed (the neighbor size)
            
            % initiate a new neuron map (without connection)
            obj.IniLayer(whichlayer);
            
            %
            Slocx = [];
            Slocy = [];
            Sweight = [];
            SWL = []; % weightlabel for sparse weights matrix
            premapsize = obj.structure{whichlayer-1}{1};
            prenmap = length(obj.structure{whichlayer-1});
            prenrow = premapsize(1);
            prencol = premapsize(2);
            Sncol = prenrow*prencol*prenmap+1; % weight matrix ncol
            
            mapsize = obj.structure{whichlayer}{1};
            nrow = mapsize(1);
            ncol = mapsize(2);
            nmap = length(obj.structure{whichlayer});
            Snrow = nrow*ncol*nmap+1; % weight matrix nrow
            % always on neuron location
            BiasN = size(obj.in{whichlayer-1},1); % the last one
            
            for whichmap = 1:nmap
                
                neighborsize = obj.config{whichlayer}{whichmap};
                neighrow = neighborsize(1);
                neighcol = neighborsize(2);
                
                premap = whichmap; %which previous maps.
                
                
                
                
                % make connection
                d = premap;
                for a = 1:nrow
                    % x, y is the position of the center of its receptive
                    % field in the previous map
                    top = (a-1)*neighrow+1;
                    down = a*neighrow;
                    for b = 1:ncol
                        left = (b-1)*neighcol+1;
                        right = b*neighcol;
                        
                        [x,y] = meshgrid(top:down,left:right); % location of pre neurons
                        x = x';
                        y = y';
                        x = x(:);
                        y = y(:);
                        z = ones(size(x))*d;
                        loc = [x,y,z];
                        
                        % initiate the connection
                        obj.Add2ConnList(whichlayer,whichmap,a,b,loc);
                        
                        %
                        xnum = a+nrow*(b-1)+nrow*ncol*(whichmap-1);
                        ytemp = x+prenrow*(y-1)+prenrow*prencol*(z-1);
                        xtemp = xnum*ones(size(ytemp));
                        % add the connection to the bias neuron
                        xtemp = [xtemp;xnum];
                        ytemp = [ytemp;BiasN];
                        
                        Slocx = [Slocx;xtemp];
                        Slocy = [Slocy;ytemp];
                        
                        % initial weights for the connections
                        L = -2.4/length(xtemp);
                        U = 2.4/length(xtemp);
                        wtemp = obj.WRand(size(ytemp),L,U);
                        % label weights for weights sharing
                        len = 2;
                        labelstart = len*(whichmap-1);
                        weightlabel = (1+labelstart)*ones(size(wtemp));
                        weightlabel(end) = 2+labelstart;
                        
                        Sweight = [Sweight;wtemp];
                        SWL = [SWL;weightlabel];
                    end
                end
            end
            obj.weights{whichlayer} = sparse(Slocx,Slocy,Sweight,Snrow,Sncol);
            obj.windex{whichlayer} = [Slocx,Slocy];
            
            obj.wshare{whichlayer} = SWL;
            
        end
        %% add a fully connected map
        function obj = AddFull(obj,whichlayer,type)
            % whichlayer is a number indicating the layer
            % whichmap indicates which map you want to generate
            % The full connected map is connected to all the neurons in the
            % previous layer
            % type=0 means with regular sigmoid activation function
            % type=1 means with other activation functions.
            narginchk(2,3);
            if nargin == 2
                type = 0;
            end
            
            premap = 1:length(obj.structure{whichlayer-1});
            
            %
            Slocx = [];
            Slocy = [];
            Sweight = [];
            premapsize = obj.structure{whichlayer-1}{1};
            prenmap = length(obj.structure{whichlayer-1});
            prenrow = premapsize(1);
            prencol = premapsize(2);
            Sncol = prenrow*prencol*prenmap+1; % weight matrix ncol
            
            mapsize = obj.structure{whichlayer}{1};
            nrow = mapsize(1);
            ncol = mapsize(2);
            nmap = length(obj.structure{whichlayer});
            Snrow = nrow*ncol*nmap+1; % weight matrix nrow
            % always on neuron location
            BiasN = size(obj.in{whichlayer-1},1); % the last one
            
            % initiate a new neuron map (without connection)
            obj.IniLayer(whichlayer,type);
            if type == 1 % type == 1, diffout is not assigned
                obj.diffout{whichlayer} = zeros(Snrow,Sncol);
                obj.diffout2{whichlayer} = zeros(Snrow,Sncol);
            end
            
            % make connection
            for whichmap = 1:nmap
                for a = 1:nrow
                    for b = 1:ncol
                        L = -2.4/length(obj.out{whichlayer-1});
                        U = 2.4/length(obj.out{whichlayer-1});
                        for d = premap
                            
                            S = obj.structure{whichlayer-1}{d}; %tempmap is a neuron object array
                            top = 1;
                            down = S(1);
                            left = 1;
                            right = S(2);
                            [x,y] = meshgrid(top:down,left:right); % location of pre neurons
                            x = x';
                            y = y';
                            x = x(:);
                            y = y(:);
                            z = ones(size(x))*d;
                            loc = [x,y,z];
                            
                            
                            % initiate the connection
                            obj.Add2ConnList(whichlayer,whichmap,a,b,loc);
                            
                            xnum = a+nrow*(b-1)+nrow*ncol*(whichmap-1);
                            ytemp = x+prenrow*(y-1)+prenrow*prencol*(z-1);
                            xtemp = xnum*ones(size(ytemp));
                            
                            
                            Slocx = [Slocx;xtemp];
                            Slocy = [Slocy;ytemp];
                            
                            % initial weights for the connections
                            wtemp = obj.WRand(size(ytemp),L,U);
                            Sweight = [Sweight;wtemp];
                            
                        end
                        % add the connection to the bias neuron
                        if type == 0
                            Slocx = [Slocx;xnum];
                            Slocy = [Slocy;BiasN];
                            wtemp = obj.WRand([1 1],L,U);
                            Sweight = [Sweight;wtemp];
                        end
                    end
                end
            end
            obj.weights{whichlayer} = sparse(Slocx,Slocy,Sweight,Snrow,Sncol);
            obj.windex{whichlayer} = [Slocx, Slocy];
%             obj.w_Grad{whichlayer} = spalloc(Snrow,Sncol,length(Slocx));
        end
        %% Update in property for one map (basic for FeedFor)
        function obj = CalLayerIn(obj,whichlayer)
            obj.in{whichlayer} = obj.weights{whichlayer}*obj.out{whichlayer-1};
        end
        %% Update out and diffout property for one map
        function obj = CalLayerout(obj,whichlayer)
            func = obj.act{whichlayer};
            difffunc = obj.diffact{whichlayer};
            
            obj.out{whichlayer} = ...
                func(obj.in{whichlayer});
            % bias neuron is always on
            obj.out{whichlayer}(end) = 1;
            
            obj.diffout{whichlayer} = ...
                difffunc(obj.in{whichlayer});
            % bias neuron doesn't need diffout
            obj.diffout{whichlayer}(end) = 0;
        end
        %% Update the RBF layer (in, out and diffout)
        function obj = CalRBF(obj,whichlayer)
            % calculate the RBF layer. label is a scalar
            % assign cache, since RBF layer is fully connected.
            mapsize = obj.structure{whichlayer}{1};
            nrow = mapsize(1);
            
            for a = 1:nrow
                obj.out{whichlayer}(a) = sum(...
                    (obj.out{whichlayer-1}(1:end-1)-obj.weights{whichlayer}(a,1:end-1)').^2);
                obj.diffout{whichlayer}(a,1:end-1) = 2*...
                    (obj.out{whichlayer-1}(1:end-1)-obj.weights{whichlayer}(a,1:end-1)')';
            end
            % bias neuron
            obj.out{whichlayer}(end) = 1;
            obj.diffout{whichlayer}(:,end) = 0;
        end
        %% Update the Loss layer
        function obj = CalLoss(obj,whichlayer,label)
            % the Loss function layer is gonna be the last layer with one
            % map only which has only one neuron
            % Label is the label for the input data. In the Lenet5 setting
            % the label is the number of the neuron in the last second
            % layer. label = 1 stands for the second neuron and etc.
            mapsize = obj.structure{whichlayer}{1};
            nneuron = length(obj.in{whichlayer-1})-1;
            if ~isequal(mapsize,[1,1])
                error('the size of the Loss layer is not correct');
            end
            temp = -obj.jconst;
            for i = 1:nneuron
                if i~=label+1
                    temp(end+1) = -obj.out{whichlayer-1}(i);
                end
            end
            lsum = logsum(temp);
            obj.out{whichlayer}(1) = ...
                obj.out{whichlayer-1}(label+1) + lsum;
            % bias neuron
            obj.out{whichlayer}(end) = 1;
            
            % Calculate the derivative of the Loss func.
            for i = 1:nneuron
                if i == label+1
                    obj.diffout{whichlayer}(1,i) = 1;
                else
                    obj.diffout{whichlayer}(1,i) = -exp(-obj.out{whichlayer-1}(i)...
                        -lsum);
                end
            end
            % bias neuron
            obj.diffout{whichlayer}(end,:) = 0;
        end
        %% Calculate the Second derivative (diffout2) for Loss layer
        function obj = D2Loss(obj,whichlayer,label)
            % The function depends on the update of the diffout of the Loss
            % layer
            mapsize = obj.structure{whichlayer}{1};
            if ~isequal(mapsize,[1,1])
                error('the size of the Loss layer is not correct');
            end
            % Calculate the derivative of the Loss func.
            obj.diffout2{whichlayer} = -obj.diffout{whichlayer}.^2-...
                obj.diffout{whichlayer};
            obj.diffout2{whichlayer}(1,label+1) = 0;
        end
        %% Calculate diffout2 for RBF layer
        function obj = D2RBF(obj,whichlayer)
            % This function calculate the diffout2 for RBF layer
            obj.diffout2{whichlayer} = diag(obj.diffout2{whichlayer+1}(1,:))*... % only one neuron in the Loss Layer
                obj.diffout{whichlayer}.^2+...
                2*diag(obj.diffout{whichlayer+1}(1,:))*ones(size(obj.diffout{whichlayer}));
%             +...
%                 2*diag(obj.diffout{whichlayer+1}(1,:))*ones(size(obj.diffout{whichlayer}));
%             
%             
%             obj.diffout2{whichlayer} = obj.diffout{whichlayer}.^2.*...
%                 repmat(obj.diffout2{whichlayer+1}(:),1,ncol)+...
%                 2*repmat(obj.diffout{whichlayer+1}(:),1,ncol);
        end
        %% Initial weights sharing methods (make the weights share the value)
        function obj = ShareWeight_Ini(obj)
            nlayer = length(obj.structure);
            for d = 1:nlayer
                if ~isempty(obj.wshare{d}) % need weight sharing
                    nlabel = size(obj.wshare{d},2);
                    for i = 1:nlabel
                        row = obj.windex{d}(obj.wshare{d}(:,i),1);
                        col = obj.windex{d}(obj.wshare{d}(:,i),2);
                        temp = obj.weights{d}(row(1),col(1));
                        nrow = size(obj.weights{d},1);
                        idx = row+(col-1)*nrow;
                        obj.weights{d}(idx) = temp;
                    end
                end
            end
        end
        %% convert wshare into a idx matrix
        function obj = Convertwshare(obj)
            nlayer = length(obj.structure);
            for d = 1:nlayer
                temp = [];
                nlabel = max(obj.wshare{d});
                for i = 1:nlabel
                    temp(:,i) = obj.wshare{d}==i;
                end
                obj.wshare{d} = sparse(logical(temp));
            end
        end
    end
    
    methods(Access = 'public') % Backpropagation Related mehtods
        %% Back weights
        function obj = CalBackweights(obj)
            nlayer = length(obj.structure);
            for d = 1:nlayer
                if isequal(obj.act{d},@CalRBF) || isequal(obj.type{d},'L')
                    [row,col] = find(obj.weights{d});
                    nrow = size(obj.weights{d},1);
                    ncol = size(obj.weights{d},2);
                    obj.backweights{d} = sparse(col,row,ones(size(col)),ncol,nrow,length(col));
                else
                    obj.backweights{d} = [];
                end
            end
        end
        %% Back Normal layer
        function obj = BackLayer(obj,whichlayer)
            temp = obj.weights{whichlayer}'*obj.diffout{whichlayer};
            ncol = size(obj.diffout{whichlayer-1},2);
%             if ncol == 1
            obj.diffout{whichlayer-1} = obj.diffout{whichlayer-1}.*temp;
%             else
%                 obj.diffout{whichlayer-1} = obj.diffout{whichlayer-1}.*repmat(temp,1,ncol);
%             end
        end      
        %% Back RBF layer and Loss layer (Full connected spec. layer)
        function obj = BackSpec(obj,whichlayer)
            if size(obj.weights{whichlayer},2) ~= size(obj.diffout{whichlayer},2)
                error('the layer is not full connected speical diffout layer');
            end
            for i = 1:size(obj.backweights{whichlayer},1)
                temp(i) = obj.backweights{whichlayer}(i,:)*...
                    obj.diffout{whichlayer}(:,i);
            end
            temp = temp(:);
            ncol = size(obj.diffout{whichlayer-1},2);
            obj.diffout{whichlayer-1} = obj.diffout{whichlayer-1}.*repmat(temp,1,ncol);
        end
        %% Back Normal Layer for second derivative
        function obj = BackLayer2(obj,whichlayer)
            temp = (obj.weights{whichlayer}.^2)'*obj.diffout2{whichlayer};
            ncol = size(obj.diffout2{whichlayer-1},2);
%             if ncol == 1
                obj.diffout2{whichlayer-1} = (obj.diffout{whichlayer-1}.^2).*temp;
%             else
%                 obj.diffout2{whichlayer-1} = (obj.diffout{whichlayer-1}.^2).*repmat(temp,1,ncol);
%             end
        end
        %% Back RBF layer for second derivative (RBF layer is fully connected)
        function obj = BackRBF2(obj,whichlayer)
            if size(obj.weights{whichlayer},2) ~= size(obj.diffout{whichlayer},2)
                error('the layer is not full connected speical diffout layer');
            end
            for i = 1:size(obj.backweights{whichlayer},1) %number of neurons in whichlayer
                temp(i) = obj.backweights{whichlayer}(i,:)*...
                    obj.diffout2{whichlayer}(:,i);
            end
            temp = temp(:);
%             ncol = size(obj.diffout{whichlayer-1},2);
            obj.diffout2{whichlayer-1} = obj.diffout{whichlayer-1}.^2.*temp;
        end
        %% Initiate w_Grad
        function obj = Iniw_Grad(obj)
            nlayer = length(obj.structure);
            for d = 1:nlayer
                if isequal(obj.act{d},@CalRBF) % RBF layer
                    obj.w_Grad{d} = zeros(size(-obj.diffout{d}));
                    obj.w_Hess{d} = zeros(size(-obj.diffout{d}));
                    obj.step{d} = zeros(size(-obj.diffout{d}));
                elseif ~isequal(obj.act{d},@CalRBF) && ~strcmp(obj.type{d},'L') % Normal lyaer
                    if isempty(obj.windex{d})
                        obj.w_Grad{d} = [];
                        obj.w_Hess{d} = [];
                        obj.step{d} = [];
                    else
                        obj.w_Grad{d} = zeros(size(obj.windex{d}(:,1)));
                        obj.w_Hess{d} = zeros(size(obj.windex{d}(:,1)));
                        obj.step{d} = zeros(size(obj.windex{d}(:,1)));
                    end
                end  
            end
        end
        %% Calculate gradient of the weights in normal layer
        function obj = GradLayer(obj,whichlayer)
            row = obj.windex{whichlayer}(:,1);
            col = obj.windex{whichlayer}(:,2);     
            %
            obj.ConnCheck(whichlayer);
            obj.NeuronCheck(whichlayer);
            value = obj.diffout{whichlayer}(row).*...
                obj.out{whichlayer-1}(col);           
%             nrow = size(obj.w_Grad{whichlayer},1);
%             ncol = size(obj.w_Grad{whichlayer},2);
            obj.w_Grad{whichlayer} = value;
%             temp = obj.diffout{whichlayer}*obj.out{whichlayer-1}';
%             temp = sparse(temp.*obj.backweights{whichlayer}');
%             obj.w_Grad{whichlayer} = temp;
        end
        %% Calculate Hessian of the error on weights in normal layer
        function obj = HessLayer(obj,whichlayer)
            row = obj.windex{whichlayer}(:,1);
            col = obj.windex{whichlayer}(:,2);     
            %
            obj.ConnCheck(whichlayer);
            obj.NeuronCheck(whichlayer);
            value = obj.diffout2{whichlayer}(row).*...
                obj.out{whichlayer-1}(col).^2;           
            obj.w_Hess{whichlayer} = value;
        end
        %% Calculate gradient of the weights in RBF layer
        function obj = GradRBF(obj,whichlayer)
            if ~isequal(obj.act{whichlayer},@CalRBF)
                error('Is not an RBF layer');
            end
            obj.w_Grad{whichlayer} = -obj.diffout{whichlayer};
        end
        %% Calculate Hessian of the weights in RBF layer
        function obj = HessRBF(obj,whichlayer)
            % D2RBF ---> HessRBF
            if ~isequal(obj.act{whichlayer},@CalRBF)
                error('Is not an RBF layer');
            end
            obj.w_Hess{whichlayer} = obj.diffout2{whichlayer};
        end
        %% Update Step
        function obj = CalStep(obj,data,label)
            nsample = size(data,1);
            nlayer = length(obj.structure);
            
            n = 1;
            obj.FeedFor(data(n,:)',label(n),1); %feed for with second derivative
            obj.BackProp(1); % Back with second derivative
            % initiate w_Hess
            for d = 1:length(obj.structure)
                w_Hess{d} = obj.w_Hess{d};
            end
            for n = 2:nsample
                obj.FeedFor(data(n,:)',label(n),1); %feed for with second derivative
                obj.BackProp(1); % Back with second derivative
%                 if n == 1
%                     % initiate w_Hess
%                     for d = 1:length(obj.structure)
%                         w_Hess{d} = zeros(size(obj.w_Hess{d}));
%                     end
%                 end
                for d = 1:nlayer
                    w_Hess{d} = w_Hess{d}+obj.w_Hess{d};
                end
            end
            for d = 1:nlayer
                w_Hess{d} = w_Hess{d}/nsample;
            end
            % weight sharing
            w_Hess = ShareWeight_Add(obj,w_Hess);
%             w_Hess = obj.ShapeWGrad(w_Hess);
            
            
            for d = 1:nlayer
                obj.step{d} = obj.eta./(obj.mu+w_Hess{d});
            end
        end
        %% Gradient weights shareing (add)
        function w_Grad = ShareWeight_Add(obj,w_Grad)
            % Perform weights sharing (w_Grad should be a cell array of 
            % vectors not a sparse matrix for most layers);
            nlayer = length(w_Grad);
            for d = 1:nlayer
                if ~isempty(obj.wshare{d}) % need weight sharing
                    nlabel = size(obj.wshare{d},2);
                    temp = obj.wshare{d}'*w_Grad{d};
%                     w_Grad{d} = obj.wshare{d}*diag(temp)*ones(size(obj.wshare{d},2),1);
                    len = length(temp);
                    w_Grad{d} = obj.wshare{d}*spdiags(temp,0,len,len)*ones(size(obj.wshare{d},2),1);
%                     for i = 1:nlabel
% %                         temp = sum(w_Grad{d}(obj.wshare{d}(:,i)));
%                         w_Grad{d}(obj.wshare{d}(:,i)) = temp(i);
%                     end
                end
            end
        end
        %% Make w_Grad the same size as weights
        function w_Grad = ShapeWGrad(obj,w_Grad)
            nlayer = length(w_Grad);
            for d = 1:nlayer
                if ~isempty(w_Grad{d})
                    row = obj.windex{d}(:,1);
                    col = obj.windex{d}(:,2);
                    nrow = size(obj.weights{d},1);
                    ncol = size(obj.weights{d},2);
                    if size(w_Grad{d},2)==1
                        w_Grad{d} = sparse2(row,col,w_Grad{d},nrow,ncol,length(row));
                    end
                end
            end
        end
        %% Update weights by using w_Grad
        function obj = UpdateWeights(obj,w_Grad)
            nlayer = length(obj.structure);
            % Perform gradient descend
            temp = cell(1,nlayer);
            for d = 1:nlayer
                if isempty(obj.w_Grad{d})
                    temp{d} = [];
                else
                    temp{d} = obj.step{d}.*w_Grad{d};
                end
            end
            temp = obj.ShapeWGrad(temp);
            for d = 2:nlayer-1
                obj.weights{d} = obj.weights{d}-temp{d};
            end
        end
    end
    
    methods % Display and Check Methods
        %% Show the network (with heat map)
        function NetShow(obj)
            nlayer = length(obj.structure);
            for d = 1:nlayer
                nmap = length(obj.structure{d});
                for c = 1:nmap
                    obj.MapShow(d,c);
                    title(strcat('layer ',num2str(d),...
                        ' map ',num2str(c)));
                    pause
                end
            end
        end
        %% Show a specific map (with heatmap)
        function MapShow(obj,whichlayer,whichmap)
            mapsize = obj.structure{whichlayer}{whichmap};
            nrow = mapsize(1);
            ncol = mapsize(2);
            from = 1+(whichmap-1)*nrow*ncol;
            to = whichmap*nrow*ncol;
            temp = obj.out{whichlayer}(from:to);
            temp = reshape(temp,[nrow,ncol]);
            imagesc(temp);
            axis equal;
        end
        %% Check the network
        function NetCheck(obj)
            obj.Calnneuron;
            obj.Calnconn;
            obj.Calntrain;
            nlayer = length(obj.structure);
            % check network size
            for d = 1:nlayer
                if isequal(obj.type{d},'N') % input layer
                    if d ~= 1
                        error('the input layer should be the first layer');
                    end
                elseif isequal(obj.type{d},'C')
                    if d == 1
                        error('the first layer should not be a convolutional layer');
                    end
                    premapsize = obj.structure{d-1}{1};
                    prenrow = premapsize(1);
                    prencol = premapsize(2);
                    
                    mapsize = obj.structure{d}{1};
                    nrow = mapsize(1);
                    ncol = mapsize(2);
                    
                    neighborsize = obj.config{d}{1}.neighbor;
                    neighrow = neighborsize(1);
                    neighcol = neighborsize(2);
                    if prenrow ~= nrow+neighrow-1 || prencol ~= ncol+neighcol-1
                        error(strcat('the size of convolutional layer ',num2str(d),' is not proper'));
                    end
                elseif isequal(obj.type{d},'S')
                    if d == 1
                        error('the first layer should not be a pulling layer');
                    end
                    premapsize = obj.structure{d-1}{1};
                    prenrow = premapsize(1);
                    prencol = premapsize(2);
                    
                    mapsize = obj.structure{d}{1};
                    nrow = mapsize(1);
                    ncol = mapsize(2);
                    
                    neighborsize = obj.config{d}{1};
                    neighrow = neighborsize(1);
                    neighcol = neighborsize(2);
                    if mod(prenrow,neighrow) ~= 0 || mod(prencol,neighcol) ~= 0
                        error(strcat('the neighborsize of the pulling ',num2str(d),' layer is not proper'));
                    end
                    if nrow ~= prenrow/neighrow || ncol ~= prencol/neighcol
                        error(strcat('the mapsize of the pulling layer ',num2str(d),' is not proper'));
                    end
                elseif isequal(obj.type{d},'F')
                    if d == 1
                        error('the first layer should not be a fully layer');
                    end
                    if isequal(obj.act{d},@CalRBF)
                        if obj.nconnection(d) ~= (obj.nneuron(d-1)-1)*(obj.nneuron(d)-1)
                            error(strcat('the RBF layer ',num2str(d),' is not fully connected'));
                        end
                    else
                        if obj.nconnection(d) ~= obj.nneuron(d-1)*(obj.nneuron(d)-1)
                            error(strcat('the F layer ',num2str(d),' is not fully connected'));
                        end
                    end
                elseif isequal(obj.type{d},'L')
                    if d == 1
                        error('the first layer should not be a Loss layer');
                    end
                    if obj.nneuron(d)-1~=1
                        error('the number of neurons in the Loss layer should be 1');
                    end
                end
            end
        end
        %% Check neuron properties
        function NeuronCheck(obj,whichlayer)
            if length(obj.in{whichlayer}) ~= obj.nneuron(whichlayer)
                warning('the size of property "in" is invalid');
                pause
            end
            if length(obj.out{whichlayer}) ~= obj.nneuron(whichlayer)
                warning('the size of property "out" is invalid');
                pause
            end
            if size(obj.diffout{whichlayer},1) ~= obj.nneuron(whichlayer)
                warning('the size of property "in" is invalid');
                pause
            end
        end
        %% Calculate network performance
        function [loss,error] = NetPerform(obj,disp)
            loss = 0;
            error = 0;
            Tset = obj.trainset;
            L = obj.label;
            nsample = size(Tset,1);
            nlayer = length(obj.structure);
            for j = 1:nsample
                data = Tset(j,:)';
                obj.FeedFor(data,L(j));
                loss = loss+obj.out{nlayer}(1,1);
                
                temp = obj.out{nlayer-1};
                temp = temp(1:2);
                [~,lab] = min(temp);
                lab = lab-1;
                if lab ~= L(j);
                    error = error + 1;
                end
            end
            loss = loss/nsample;
            error = error/nsample;
            if disp
                display(loss);
                display(error);
            end
        end
        %% Ouptut RBK layer probability
        function prob = CalProb(obj,data)
            % observations are row wise in the data matrix
            nsample = size(data,1);
            prob = zeros(nsample,1);
            RBKlayer = length(obj.structure)-1;
            parfor i = 1:nsample
                obj.FeedFor(data(i,:),0);
                temp = exp(obj.out{RBKlayer});
                prob(i) = temp(2)/sum(temp);
            end
        end
        
        %% Check Gradient
        function [error,loc] = GradCheck(obj,data,label)
            percent = 0.03;
            error = [];
            loc = [];
            pert = 0.0001;
            for d = 1:8
                if obj.nconnection(d)<20 && obj.nconnection(d)>0
                    idx{d} = 1:obj.nconnection(d);
                else
                    idx{d} = randsample(obj.nconnection(d),floor(percent*obj.nconnection(d)));
                end
                for i = 1:length(idx{d})
                    w = obj.weights{d};
                    
                    x = obj.windex{d}(idx{d}(i),1);
                    y = obj.windex{d}(idx{d}(i),2);
                    
                    obj.FeedFor(data,label);
                    obj.BackProp;
                    output1 = obj.out{9}(1);
                    if d ~= 8
                        g_true = obj.w_Grad{d}(idx{d}(i));
                    elseif d == 8
                        g_true = obj.w_Grad{d}(x,y);
                    end
                    obj.weights{d}(x,y) = obj.weights{d}(x,y)+pert;
                    obj.FeedFor(data,label);
                    output2 = obj.out{9}(1);
                    g = (output2-output1)/pert;
                    if abs(g-g_true)>1e-5
                        error(end+1) = g-g_true;
                        loc(end+1,1) = x;
                        loc(end,2) = y;
                        loc(end,3) = d;
                    end
                    obj.weights{d} = w;
                end
            end
            error = error';
        end
        %% numerical Gradient
        function [simG,trueG] = NumGrad(obj,data,label,layer,x,y)
            pert = 0.0001;
            w = obj.weights{layer};
            
            obj.FeedFor(data,label);
            obj.BackProp;
            temp = obj.ShapeWGrad(obj.w_Grad);
            trueG = temp{layer}(x,y);
            output1 = obj.out{9}(1);
            
            obj.weights{layer}(x,y) = obj.weights{layer}(x,y)+pert;
            obj.FeedFor(data,label);
            output2 = obj.out{9}(1);
            simG = (output2-output1)/pert;
            
            obj.weights{layer} = w;
        end
        %% numerical Hessian
        function [simH,trueH] = NumHess(obj,data,label,layer,x,y)
            pert = 0.0001;
            w = obj.weights{layer};
            
            obj.FeedFor(data,label,1);
            obj.BackProp(1);
            temp = obj.ShapeWGrad(obj.w_Hess);
            trueH = temp{layer}(x,y);
            fx = obj.out{9}(1);
            
            obj.weights{layer}(x,y) = obj.weights{layer}(x,y)+pert;
            obj.FeedFor(data,label);
            fxplus = obj.out{9}(1);
            
            obj.weights{layer}(x,y) = obj.weights{layer}(x,y)-2*pert;
            obj.FeedFor(data,label);
            fxminus = obj.out{9}(1);
            
            simH = (fxplus+fxminus-2*fx)/(pert^2);
            
            obj.weights{layer} = w;
        end
        %% check the connections
        function obj = ConnCheck(obj,whichlayer)
            row = obj.windex{whichlayer}(:,1); % post syn
            col = obj.windex{whichlayer}(:,2); % pre syn
            row = max(row);
            col = max(col);
            if row > obj.nneuron(whichlayer)
                warning(strcat('connection post index exceeds the neuron in layer ',num2str(whichlayer),'. ',...
                    num2str(row)));
                pause
            end
            if col > obj.nneuron(whichlayer-1)
                warning(strcat('connection pre index exceeds the neuron in layer ',num2str(whichlayer),'. ',...
                    num2str(col)));
                pause
            end
        end
        %% Calculate the number of connections in each layer
        function obj = Calnconn(obj)
            nlayer = length(obj.structure);
            for d = 1:nlayer
                obj.nconnection(d) = nnz(obj.weights{d});
            end
        end
        %% Calculate the number of neurons in each layer
        function obj = Calnneuron(obj)
            nlayer = length(obj.structure);
            for d = 1:nlayer
                obj.nneuron(d) = length(obj.out{d});
            end
            obj.nneuron = obj.nneuron;
        end
        %% Calculate the number of trainable parameters in each layer
        function obj = Calntrain(obj)
            nlayer = length(obj.structure);
            for d = 1:nlayer
                if isempty(obj.wshare{d})
                    obj.ntrain(d) = obj.nconnection(d);
                else
                    obj.ntrain(d) = size(obj.wshare{d},2)+obj.nconnection(d)-nnz(obj.wshare{d});
                end
            end
        end
    end
    
    methods(Static)
        %% display a single data point (one hand-written digit)
        function ShowOne(input)
            % input is a row vector whose length should be a squared number
            len = length(input);
            len = sqrt(len);
            temp = reshape(input,len,len);
            temp = temp';
            imagesc(temp)
            colormap gray;
        end
        %% randomly initiate weights
        function output = WRand(size,low,up)
            if up < low
                error('up should be bigger than low');
            end
            output = low + (up-low).*rand(size);
        end
        %% patch image extend 28by28 to 32by32
        function output = PatchIm(im)
            % im is a vector
            im = reshape(im,[28,28]);
            output = zeros(32,32);
            output(3:30,3:30) = im;
            output = output(:);
        end
    end
    
end