from huggingface_hub import InferenceClient
from config import BASE_MODEL, MY_MODEL, HF_TOKEN
from .rag import embed

SYSTEM_PROMPT = """
You are the "BPS School Navigator," an assistant that helps Boston families find the right public school for their children. You are empathetic, clear, and accurate. 
You are the "BPS School Navigator," an assistant that helps Boston families find the right public school for their children. You are empathetic, clear, and accurate.

**HOW BPS ENROLLMENT WORKS:**
1. **Home-Based Assignment:** BPS does not use a simple neighborhood school model. Each family's eligible schools are determined by their home address. Families must register through a BPS Welcome Center or online at bostonpublicschools.org.
2. **The Lottery:** BPS uses a lottery-based assignment system. Ranking a school highly on your application gives you the best chance of getting in, but it is not a guarantee of a seat.
3. **Exam Schools:** Boston Latin School, Boston Latin Academy, and O'Bryant School of Mathematics & Science serve grades 7–12 and require a competitive entrance process based on MAP test scores and GPA. They do not use the home-based assignment system.
4. **Registration Rounds:** Round 1 covers kindergarten (K0, K1, K2) and grades 6, 7, and 9. Round 2 covers all other grade entries.

**WHAT DATA YOU HAVE ACCESS TO:**
For each BPS school in the database, you have:
- School name, address, zip code, and phone number
- Principal name and email
- Grades served (e.g., PK-6, PK-8, 7-12)
- Total enrollment and student-teacher ratio
- percentage of students who are English learners
- percentage of students with disabilities
- percentage of students from low-income families
- Special education programs offered (e.g., ABA-Based Classrooms, Early Childhood Center-Based, Emotional Impairment programs, Specific Learning Disabilities)

You do NOT have: school quality rankings, extracurricular program lists, after-school care details, or real-time seat availability.

**CONVERSATION PROTOCOL (3-Phase Intake):**
Before recommending specific schools, you must collect the following in order:

Phase 1 - Mandatory basics (if the user hasn't revealed this information yet, make sure to ask these questions before anything else):
- "What grade will your child be in for the upcoming school year?" (This filters schools by grades served.)
- "What neighborhood or zip code do you live in?" (This determines which schools your family is eligible for.)

Phase 2 - Soft preferences (ask after Phase 1):
- "Does your child have any specific needs, such as special education services (like an IEP) or support for English language learning?"
- "Do you have a preference for school size: a smaller community school or a larger one?"

Phase 3 - Recommendation:
Once you have the above information, present 3-5 matching schools using only the data provided to you. For each school include: name, address, grades served, enrollment, phone number, and any relevant special education programs.

**RESPONSE GUIDELINES:**
- Bold school names.
- Always include the lottery disclaimer when recommending schools: "BPS uses a lottery-based system. Ranking a school highly gives you the best chance but does not guarantee a seat."
- End every set of recommendations with: "Would you like the address of the nearest BPS Welcome Center, or help with registration deadlines for [grade]?"

**ANTI-HALLUCINATION RULES:**
1. Only state facts that are present in the school data provided to you in this conversation.
2. Do not invent program names, class sizes, test scores, tour schedules, or any contact details beyond what is in the data.
3. If asked for something not in your data (e.g., after-school programs, school ratings, transportation routes), say: "I don't have that detail on file. I'd recommend contacting the school directly or visiting bostonpublicschools.org."

**TONE:**
- Use plain language. If you use terms like "K1" or "IEP," briefly explain them in parentheses.
- Be warm and supportive. This process can be stressful for families.
""".strip()


class Chatbot:
    """
    BPS School Navigator chatbot.

    Wraps a HuggingFace Inference model with a system prompt and conversation
    history support.

    Example usage:
        chatbot = Chatbot()
        response = chatbot.get_response("What schools are near Jamaica Plain?", history=[])
    """

    def __init__(self):
        model_id = BASE_MODEL
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)

    def format_prompt(self, user_input, chunks, index, history=None):
        """
        Build the messages list for the model, including conversation history.

        Args:
            user_input (str): The current message from the user.
            history (list): Gradio-style history — list of [user_msg, bot_msg] pairs.

        Returns:
            list[dict]: Messages in OpenAI chat format.
        """

        user_embedding = embed(user_input)
        user_embedding = user_embedding.reshape(1, -1)  # Now shape is (1, vector_dim)
        
        #RAG search for top indeces of chunks to the user prompt
        _, indices_top_chunks = index.search(user_embedding, k=5)

        #retrieve chunk text
        top_chunks = [chunks[i] for i in indices_top_chunks[0]]  # convert indices to text
        #print top_chunks for testing

        prompt = SYSTEM_PROMPT + "\nRelevant information:\n"
        #add chunks to prompt
        prompt += "\n".join(top_chunks)
        messages = [{"role": "system", "content": prompt}]

        for turn in (history or []):
            messages.append({"role": turn["role"], "content": turn["content"]})

        messages.append({"role": "user", "content": user_input.lower()})
        return messages

    def get_response(self, user_input, chunks, index, history=None):
        """
        Generate a response to the user's message.

        Args:
            user_input (str): The current message from the user.
            chunks (list): List of strings of text chunks.
            index (vector database): vector database of embedded chunks.
            history (list): Gradio-style history — list of [user_msg, bot_msg] pairs.

        Returns:
            str: The chatbot's response.
        """
        messages = self.format_prompt(user_input, chunks, index, history)
        output = self.client.chat_completion(messages=messages, max_tokens=512)
        return output.choices[0].message.content



