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