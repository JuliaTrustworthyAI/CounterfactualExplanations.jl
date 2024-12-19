"Abstract type for serializers."
abstract type AbstractSerializer end

"Standard serializer (allows serialization)."
struct Serializer <: AbstractSerializer end

"Serializer that allows serialization."
SerializationState(serializer::Serializer) = true

"Null serializer (does not allow serialization)."
struct NullSerializer <: AbstractSerializer end

"Serializer that does not allow serialization."
SerializationState(serializer::NullSerializer) = false

"Global serializer state (allows or disallows serialization)."
global _serialization_state::Bool = true

"""
    global_serializer(serializer::AbstractSerializer)

Set the global serializer to `serializer` and return its state. The global serializer is used by default for all serialization operations.
"""
function global_serializer(serializer::AbstractSerializer)
    global _serialization_state = SerializationState(serializer)
    return _serialization_state
end

"Abstract type for output identifiers."
abstract type AbstractOutputIdentifier end

"Default output identifier (no specific ID)."
struct DefaultOutputIdentifier <: AbstractOutputIdentifier end

OutputID(identifier::DefaultOutputIdentifier) = ""

global _output_id::String = ""

"And explicit output identifier that takes the string value of `id`. "
struct ExplicitOutputIdentifier <: AbstractOutputIdentifier
    id::String
end

OutputID(identifier::ExplicitOutputIdentifier) = identifier.id

get_global_output_id() = _output_id

"""
    global_output_identifier(identifier::AbstractOutputIdentifier)

Set the global output identifier to `identifier` and return its string representation. The global output identifier is used by default for all serialization operations.
"""
function global_output_identifier(identifier::AbstractOutputIdentifier)
    global _output_id = OutputID(identifier)
    return _output_id
end
