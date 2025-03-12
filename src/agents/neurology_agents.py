class AnswerSynthesisAgent(BaseNeurologyAgent):
    """
    Agent that synthesizes answers based on retrieved information.
    """
    def __init__(self, config: Dict[str, Any], llm: Any, model_name: str = None):
        self.model_name = model_name
        super().__init__(config, llm)
    
    def _get_tools(self) -> List[Any]:
        return [AnswerSynthesisTool(llm=self.llm, model_name=self.model_name)]


class ValidatorOutput(BaseModel):
    original_question: str
    selected_answer: str  # a | b | c | d
    validation_result: str  # Valid or Invalid with explanation
    final_answer: str  # The potentially corrected answer

    class ConfigDict:
        extra = "allow"


class AnswerValidationTool(BaseTool):
    """
    Tool for validating synthesized answers against retrieved information.
    """
    name: str = "Answer Validation Tool"
    description: str = "Validates the synthesized answer against retrieved knowledge and ensures it is well-reasoned and supported by evidence."
    
    llm: Any = None
    model_name: str = None

    def __init__(self, llm, model_name=None):
        super().__init__()
        self.llm = llm
        self.model_name = model_name
    
    def _run(self, original_question: str, selected_answer: str, reasoning: str) -> ValidatorOutput:
        """Validates the synthesized answer and reasoning."""
        model_name_safe = self.model_name.replace(':', '-') if self.model_name else 'default-model'
        retrieved_data_file_path = f"retrieved_data_{model_name_safe}.json"
        
        try:
            # Read the retrieved knowledge from file
            with open(retrieved_data_file_path, 'r') as f:
                retrieved_data = f.read()

            system_prompt = """
You are a medical expert specializing in neurology. Your role is to validate answers to medical exam questions.
Carefully analyze the selected answer and reasoning against the retrieved context.
Determine if the answer is valid based on the following criteria:
1. Consistency with the retrieved medical knowledge
2. Sound clinical reasoning and evidence-based justification
3. Consideration of differential diagnoses and key clinical factors

If the answer appears valid, confirm it. If invalid, explain why and suggest the correct answer."""
            
            # Construct prompt for LLM
            prompt = f"""
            ## RETRIEVED CONTEXT:
            {retrieved_data}

            ## ORIGINAL QUESTION:
            {original_question}
            
            ## SELECTED ANSWER:
            {selected_answer}
            
            ## PROVIDED REASONING:
            {reasoning}
            
            Please validate this answer. Is it consistent with neurological knowledge and sound clinical reasoning?
            If valid, confirm the answer. If invalid, explain why and provide the correct answer with justification.
            
            Response format (JSON):
            {{
              "validation_result": "Valid/Invalid with explanation",
              "final_answer": "a/b/c/d (potentially corrected answer)"
            }}
            """

            # Get the validation from the LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            
            response = self.llm.chat(messages=messages, temperature=0.5)
            model_response = response.get("text", "")

            try:
                # Try parsing the response as JSON
                response_json = json.loads(model_response)
                output = ValidatorOutput(
                    original_question=original_question,
                    selected_answer=selected_answer,
                    validation_result=response_json.get('validation_result', ''),
                    final_answer=response_json.get('final_answer', selected_answer)  # Default to original if not provided
                )
                return output
                
            except json.JSONDecodeError:
                # If JSON parsing fails, create a basic response
                return ValidatorOutput(
                    original_question=original_question,
                    selected_answer=selected_answer,
                    validation_result="Validation parsing error, see raw response",
                    final_answer=selected_answer  # Keep original answer if validation fails
                )
                
        except Exception as e:
            print(f"Error validating answer: {e}")
            return ValidatorOutput(
                original_question=original_question,
                selected_answer=selected_answer,
                validation_result=f"Validation error: {str(e)}",
                final_answer=selected_answer  # Keep original answer if validation fails
            )


class ValidatorAgent(BaseNeurologyAgent):
    """
    Agent that validates answers by checking against retrieved knowledge.
    """
    def __init__(self, config: Dict[str, Any], llm: Any, model_name: str = None):
        self.model_name = model_name
        super().__init__(config, llm)
    
    def _get_tools(self) -> List[Any]:
        return [AnswerValidationTool(llm=self.llm, model_name=self.model_name)]