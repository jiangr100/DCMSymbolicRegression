module MultiFeatureNodeModule

# Import specific functions and types you need to extend
import DynamicExpressions:
    DynamicExpressions,
    AbstractNode,
    AbstractExpressionNode,
    set_node!,
    AbstractOperatorEnum,
    string_tree,
    tree_mapreduce
using DynamicExpressions.UtilsModule: Undefined, Nullable, deprecate_varmap
using DynamicExpressions.NodeModule: Node, GraphNode
using DynamicExpressions.StringsModule: 
    strip_brackets, 
    OpNameDispatcher, 
    combine_op_with_inputs,
    string_constant,
    string_variable,
    _not

# Import functions we need to extend
import DynamicExpressions.NodeModule: 
    defines_eltype,
    constructorof,
    with_type_parameters,
    max_degree, 
    has_max_degree,
    make_default,
    all_defaults,
    node_factory,
    get_child,
    set_children!,
    set_leaf!,
    leaf_copy,
    leaf_equal,
    @make_accessors

using DispatchDoctor: @unstable
using Random: AbstractRNG, default_rng, randn

# Re-export what users might need
export MultiFeatureNode

const DEFAULT_NODE_TYPE = Float64
const DEFAULT_MAX_DEGREE = 2

"""
    MultiFeatureNode{T,D} <: AbstractExpressionNode{T,D}

A leaf node that can represent either:
1. A constant value (constant = true, is_single_feature = false)
2. A single feature (constant = false, is_single_feature = true)  
3. Multiple features with coefficients (constant = false, is_single_feature = false)

# Fields

- `degree::UInt8`: Always 0 (leaf node)
- `constant::Bool`: True if this is a constant node
- `is_single_feature::Bool`: True if this is a single feature node
- `val::T`: Value for constant nodes, or unused for feature nodes
- `feature::UInt16`: Feature index for single feature nodes
- `features::Vector{UInt16}`: List of feature indices for multi-feature nodes
- `coefficients::Vector{T}`: Coefficients for multi-feature nodes (including bias term)
- `children::NTuple{D,Nullable{MultiFeatureNode{T,D}}}`: Children tuple (always empty for leaf nodes)

# Constructors

    # Constant node
    MultiFeatureNode([T]; val)
    MultiFeatureNode{T}(; val)

    # Single feature node  
    MultiFeatureNode([T]; feature::Int)
    MultiFeatureNode{T}(; feature::Int)

    # Multi-feature node with specified coefficients
    MultiFeatureNode([T]; features::Vector{Int}, coefficients::Vector)
    MultiFeatureNode{T}(; features::Vector{Int}, coefficients::Vector)
"""
mutable struct MultiFeatureNode{T,D} <: AbstractExpressionNode{T,D}
    degree::UInt8  # Always 0
    constant::Bool  # True for constant nodes
    is_single_feature::Bool  # True for single feature nodes
    val::T  # Value for constant nodes
    # ------------------- (conditionally defined based on node type)
    feature::UInt16  # For single feature nodes only
    features::Vector{UInt16}  # For multi-feature nodes only
    coefficients::Vector{T}  # For multi-feature nodes only
    # ------------------- (operator not used for leaf nodes)
    op::UInt8  # Not used
    # ------------------- (children always empty for leaf nodes)
    children::NTuple{D,Nullable{MultiFeatureNode{T,D}}}

    #################
    ## Constructors:
    #################
    MultiFeatureNode{_T,_D}() where {_T,_D} = new{_T,_D::Int}()
    MultiFeatureNode{_T}() where {_T} = MultiFeatureNode{_T,DEFAULT_MAX_DEGREE}()
end

@unstable function with_max_degree(::Type{<:MultiFeatureNode}, ::Val{D}) where {D}
    return MultiFeatureNode{T,D} where {T}
end

@unstable function with_default_max_degree(::Type{<:MultiFeatureNode})
    return with_max_degree(MultiFeatureNode, Val(DEFAULT_MAX_DEGREE))
end

function default_allocator(::Type{<:MultiFeatureNode}, ::Type{T}) where {T}
    return with_type_parameters(MultiFeatureNode, T)()
end

preserve_sharing(::Union{Type{<:MultiFeatureNode},MultiFeatureNode}) = false

# Make accessors work with .l and .r syntax
@make_accessors MultiFeatureNode

# Constructors for different node types
function node_factory(
    ::Type{<:MultiFeatureNode}, ::Type{T1}, val::T2, ::Nothing, ::Nothing, ::Nothing, allocator::F
) where {T1,T2,F}
    # Constant node
    T = node_factory_type(MultiFeatureNode, T1, T2)
    n = allocator(MultiFeatureNode, T)
    n.degree = 0
    n.constant = true
    n.is_single_feature = false
    n.val = convert(T, val)
    return n
end

function node_factory(
    ::Type{<:MultiFeatureNode}, ::Type{T1}, ::Nothing, feature::Integer, ::Nothing, ::Nothing, allocator::F
) where {T1,F}
    # Single feature node
    T = node_factory_type(MultiFeatureNode, T1, DEFAULT_NODE_TYPE)
    n = allocator(MultiFeatureNode, T)
    n.degree = 0
    n.constant = false
    n.is_single_feature = true
    @assert feature >= 1
    n.feature = UInt16(feature)
    return n
end

"""Create an operator node."""
@inline function node_factory(
    ::Type{N}, ::Type, ::Nothing, ::Nothing, op::Integer, children::Union{Tuple,AbstractVector}, allocator::F,
) where {N<:MultiFeatureNode,F}
    T = defines_eltype(N) ? eltype(N) : promote_type(map(eltype, children)...)
    defines_eltype(N) && @assert T === promote_type(T, map(eltype, children)...)
    D2 = length(children)
    @assert D2 <= max_degree(N)
    NT = with_type_parameters(N, T)
    n = allocator(N, T)
    n.degree = D2
    n.op = op
    set_children!(n, children)
    return n
end

# Additional constructor for multi-feature nodes
function multi_feature_node_factory(
    ::Type{<:MultiFeatureNode}, ::Type{T1}, features::Vector{UInt16}, coefficients::Vector{T2}, allocator::F
) where {T1,T2,F}
    # Multi-feature node with specified coefficients
    T = node_factory_type(MultiFeatureNode, T1, T2)
    
    # Validate coefficients length
    if length(coefficients) != length(features) + 1
        error("MultiFeatureNode coefficients must have length = length(features) + 1. Got $(length(coefficients)) coefficients for $(length(features)) features")
    end
    
    n = allocator(MultiFeatureNode, T)
    n.degree = 0
    n.constant = false
    n.is_single_feature = false
    n.features = UInt16.(features)
    n.coefficients = convert(Vector{T}, coefficients)
    return n
end

@inline function node_factory_type(::Type{N}, ::Type{T1}, ::Type{T2}) where {N,T1,T2}
    if T1 === Undefined && !defines_eltype(N)
        T2
    elseif T1 === Undefined
        eltype(N)
    elseif !defines_eltype(N)
        T1
    else
        eltype(N)
    end
end

# Main constructor
function (::Type{N})(
    ::Type{T1}=Undefined; 
    val=nothing, 
    feature=nothing, 
    op=nothing, 
    l=nothing, 
    r=nothing, 
    children=nothing, 
    allocator::F=default_allocator,
    features=nothing,
    coefficients=nothing
) where {T1,N<:MultiFeatureNode,F}
    # Handle different node types
    if N <: MultiFeatureNode
        if val !== nothing && feature === nothing && features === nothing
            # Constant node
            return node_factory(N, T1, val, feature, op, children, allocator)
        elseif feature !== nothing && val === nothing && features === nothing
            # Single feature node
            return node_factory(N, T1, val, feature, op, children, allocator)
        elseif features !== nothing && coefficients !== nothing && val === nothing && feature === nothing
            # Multi-feature node
            return multi_feature_node_factory(N, T1, features, coefficients, allocator)
        # else
        #     error("MultiFeatureNode requires exactly one of: val (constant), feature (single feature), or features+coefficients (multi-feature)")
        end
    end
    
    _children = if !isnothing(l) && isnothing(r)
        @assert isnothing(children)
        (l,)
    elseif !isnothing(l) && !isnothing(r)
        @assert isnothing(children)
        (l, r)
    else
        children
    end
    if all_defaults(N, val, feature, op, _children)
        return make_default(N, T1)
    end
    return node_factory(N, T1, val, feature, op, _children, allocator)
end

"""
    string_tree(
        tree::AbstractExpressionNode{T},
        operators::Union{AbstractOperatorEnum,Nothing}=nothing;
        f_variable::F1=string_variable,
        f_constant::F2=string_constant,
        variable_names::Union{Array{String,1},Nothing}=nothing,
        # Deprecated
        varMap=nothing,
    )::String where {T,F1<:Function,F2<:Function}

Convert an equation to a string.

# Arguments
- `tree`: the tree to convert to a string
- `operators`: the operators used to define the tree

# Keyword Arguments
- `f_variable`: (optional) function to convert a variable to a string, with arguments `(feature::UInt8, variable_names)`.
- `f_constant`: (optional) function to convert a constant to a string, with arguments `(val,)`
- `variable_names::Union{Array{String, 1}, Nothing}=nothing`: (optional) what variables to print for each feature.
"""
function string_tree(
    tree::MultiFeatureNode{T},
    operators::Union{AbstractOperatorEnum,Nothing}=nothing;
    f_variable::F1=string_variable,
    f_constant::F2=string_constant,
    variable_names=nothing,
    pretty::Union{Bool,Nothing}=nothing, # Not used, but can be used by other types
    # Deprecated
    raw::Union{Bool,Nothing}=nothing,
    varMap=nothing,
)::String where {T,F1<:Function,F2<:Function}
    if !isnothing(raw)
        Base.depwarn("`raw` is deprecated; use `pretty` instead", :string_tree)  # COV_EXCL_LINE
    end
    pretty = @something(pretty, _not(raw), false)
    variable_names = deprecate_varmap(variable_names, varMap, :string_tree)
    raw_output = tree_mapreduce(
        let f_constant = f_constant,
            f_variable = f_variable,
            variable_names = variable_names

            (leaf,) -> if leaf.constant
                collect(f_constant(leaf.val))::Vector{Char}
            elseif leaf.is_single_feature
                collect(f_variable(leaf.feature, variable_names))::Vector{Char}
            else
                collect(multi_feature_string(leaf, variable_names))::Vector{Char}
            end
        end,
        OpNameDispatcher{max_degree(tree),typeof(operators)}(operators, pretty),
        combine_op_with_inputs,
        tree,
        Vector{Char};
        f_on_shared=(c, is_shared) -> if is_shared
            out = ['{']
            append!(out, c)
            push!(out, '}')
            out
        else
            c
        end,
    )
    return String(strip_brackets(raw_output))
end

# String representation for MultiFeatureNode
function multi_feature_string(
    tree::MultiFeatureNode,
    variable_names::Union{AbstractVector{String},Nothing}=nothing,
    f_variable::F1=string,
    f_constant::F2=string_constant
) where {F1, F2}
    @assert !tree.constant && !tree.is_single_feature
    # Multi-feature node
    terms = String[]
    
    # Add bias term first
    bias = tree.coefficients[end]
    if bias != zero(typeof(bias))
        push!(terms, f_constant(bias))
    end
    
    # Add feature terms
    for (i, (feature, coeff)) in enumerate(zip(tree.features, tree.coefficients[1:end-1]))
        var_name = if variable_names !== nothing && feature <= length(variable_names)
            variable_names[feature]
        else
            "x$(feature)"
        end
        
        if coeff != zero(typeof(coeff))
            if isempty(terms)
                # First term
                term = "$(f_constant(coeff)) * $(f_variable(var_name))"
            else
                sign = coeff >= zero(typeof(coeff)) ? " + " : " - "
                abs_coeff = abs(coeff)
                term = "$(sign)$(f_constant(abs_coeff)) * $(f_variable(var_name))"
            end
            push!(terms, term)
        end
    end
    
    return isempty(terms) ? f_constant(zero(eltype(tree.coefficients))) : ("(" * join(terms, "") * ")")
end

function set_leaf!(tree::MultiFeatureNode, new_leaf::MultiFeatureNode)
    tree.constant = new_leaf.constant
    tree.is_single_feature = new_leaf.is_single_feature
    if new_leaf.constant
        tree.val = new_leaf.val::eltype(new_leaf)
    elseif new_leaf.is_single_feature
        tree.feature = new_leaf.feature
    else
        tree.features = copy(new_leaf.features)
        tree.coefficients = copy(new_leaf.coefficients)
    end
    return nothing
end

function leaf_copy(t::N) where {T,N<:MultiFeatureNode{T}}
    if t.constant
        return constructorof(N)(; val=t.val)
    elseif t.is_single_feature
        return constructorof(N)(T; feature=t.feature)
    else
        return constructorof(N)(T; features=t.features, coefficients=t.coefficients)
    end
end

function leaf_convert(
    ::Type{N1}, t::N2
) where {T1,T2,N1<:MultiFeatureNode{T1},N2<:MultiFeatureNode{T2}}
    if t.constant
        return constructorof(N1)(T1; val=convert(T1, t.val::T2))
    elseif t.is_single_feature
        return constructorof(N)(T; feature=t.feature)
    else
        return constructorof(N)(T; features=t.features, coefficients=t.coefficients)
    end
end

@inline function leaf_equal(
    a::MultiFeatureNode{T1}, b::MultiFeatureNode{T2}
) where {T1,T2}
    constant = a.constant
    constant != b.constant && return false
    a.is_single_feature != b.is_single_feature && return false
    if constant
        return a.val::T1 == b.val::T2
    elseif a.is_single_feature
        return a.feature == b.feature
    else
        length(a.features) != length(b.features) && return false
        for i in 1:length(a.features)
            a.features[i] != b.features[i] && return false
            a.coefficients[i] != b.coefficients[i] && return false
        end
    end
end

# function leaf_hash(h::UInt, t::AbstractExpressionNode)
#     return t.constant ? hash((0, t.val), h) : hash((1, t.feature), h)
# end

# Promotion rules
function Base.promote_rule(::Type{MultiFeatureNode{T1,D}}, ::Type{MultiFeatureNode{T2,D}}) where {T1,T2,D}
    return MultiFeatureNode{promote_type(T1, T2),D}
end

function Base.promote_rule(::Type{MultiFeatureNode{T1,D}}, ::Type{Node{T2,D}}) where {T1,T2,D}
    return MultiFeatureNode{promote_type(T1, T2),D}
end

function Base.promote_rule(::Type{MultiFeatureNode{T1,D}}, ::Type{GraphNode{T2,D}}) where {T1,T2,D}
    return MultiFeatureNode{promote_type(T1, T2),D}
end

end