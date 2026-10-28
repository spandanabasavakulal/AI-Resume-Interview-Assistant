
import io
import math
import time
import requests
import streamlit as st
import fitz


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_BASE_URL = "http://localhost:11434"
CHAT_MODEL = "llama3.2:latest"
EMBEDDING_MODEL = "nomic-embed-text:latest"
REQUEST_TIMEOUT = 300

st.set_page_config(
    page_title="AI Resume & Interview Assistant",
    page_icon="🎯",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "resume_text": "",
    "job_role": "",
    "job_description": "",
    "match_result": "",
    "similarity_score": None,
    "improvement_report": "",
    "prep_report": "",
    "messages": [],
    "interview_started": False,
    "interview_finished": False,
    "evaluation_report": "",
    "interview_error": "",
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# OLLAMA CONNECTION
# ============================================================

def ask_ollama(messages, temperature=0.3):
    """Send a chat request to the locally running Ollama model."""

    url = f"{OLLAMA_BASE_URL}/api/chat"

    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    response = requests.post(
        url,
        json=payload,
        timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()

    data = response.json()
    return data["message"]["content"].strip()


def get_embedding(text):
    """Generate an embedding using the local Nomic embedding model."""

    url = f"{OLLAMA_BASE_URL}/api/embeddings"

    response = requests.post(
        url,
        json={
            "model": EMBEDDING_MODEL,
            "prompt": text
        },
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()
    return response.json()["embedding"]


def cosine_similarity(vec_a, vec_b):
    """Calculate cosine similarity between two embedding vectors."""

    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))

    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def split_text(text, max_chars=6000):
    """Limit prompt size while preserving the beginning and end of text."""

    text = text.strip()

    if len(text) <= max_chars:
        return text

    half = max_chars // 2

    return (
        text[:half]
        + "\n\n[Middle content omitted to limit prompt size]\n\n"
        + text[-half:]
    )


# ============================================================
# FAST PDF EXTRACTION
# ============================================================

@st.cache_data(show_spinner=False, max_entries=5)
def extract_pdf_text_cached(pdf_bytes):
    """
    Extract selectable text from PDF bytes.

    Caching avoids extracting the same PDF again during Streamlit reruns.
    This function does not run OCR on scanned PDFs.
    """

    text_parts = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        if len(pdf) == 0:
            raise ValueError("The uploaded PDF contains no pages.")

        for page in pdf:
            page_text = page.get_text("text")

            if page_text and page_text.strip():
                text_parts.append(page_text.strip())

    extracted_text = "\n\n".join(text_parts).strip()

    if not extracted_text:
        raise ValueError(
            "No selectable text was found. "
            "Your PDF may be scanned and require OCR."
        )

    return extracted_text


def extract_resume_text(uploaded_file):
    """Extract resume text efficiently from an uploaded PDF."""

    try:
        pdf_bytes = uploaded_file.getvalue()
        return extract_pdf_text_cached(pdf_bytes)

    except Exception as error:
        raise ValueError(f"Could not extract resume text: {error}") from error


# ============================================================
# SECTION 2: RESUME AND JOB DESCRIPTION ANALYSIS
# ============================================================

def compare_resume_to_jd(resume_text, job_description):
    """Compare resume content with the target job description."""

    resume_for_analysis = split_text(resume_text, 7000)
    jd_for_analysis = split_text(job_description, 5000)

    resume_embedding = get_embedding(
        split_text(resume_text, 8000)
    )

    jd_embedding = get_embedding(
        split_text(job_description, 8000)
    )

    similarity = cosine_similarity(
        resume_embedding,
        jd_embedding
    )

    # Similarity is not a validated ATS score or hiring probability.
    similarity_percent = round(max(0, similarity) * 100, 1)

    prompt = f"""
You are a careful resume and job-description analyst.

Analyze the candidate's resume against the job description.

IMPORTANT RULES:
- Use only information supported by the provided resume.
- Do not invent skills, projects, qualifications, or experience.
- Missing evidence does not prove the candidate lacks a skill.
- Distinguish clearly between confirmed matches and unclear evidence.
- Do not claim that the candidate will or will not be hired.

Return a structured report using these headings:

## Resume Summary
## Job Description Requirements
## Matching Skills
## Missing or Unclear Requirements
## Weakly Represented Skills
## Relevant Projects and Experience
## Overall Assessment

RESUME:
{resume_for_analysis}

JOB DESCRIPTION:
{jd_for_analysis}
"""

    report = ask_ollama(
        [{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return {
        "report": report,
        "similarity": similarity_percent
    }


# ============================================================
# SECTION 3: RESUME IMPROVEMENT
# ============================================================

def generate_resume_improvements(resume_text, job_description, job_role):
    """Generate truthful, role-specific resume improvement suggestions."""

    prompt = f"""
You are an expert resume reviewer.

Target job role: {job_role}

Review the resume against the job description.

Return a practical report with these headings:

1. Top Five Changes to Prioritize
2. Existing Resume Bullets That Could Be Rewritten
3. Keywords from the Job Description Genuinely Supported by the Resume
4. Relevant Skills to Learn if They Are Not Currently Demonstrated
5. Formatting and Clarity Suggestions

For each rewritten bullet:
- Show the original bullet.
- Show the improved bullet.
- Preserve the original meaning.
- Do not invent results, percentages, achievements, tools, or responsibilities.
- Do not claim an improvement in performance unless the original resume supports it.

Resume:
{split_text(resume_text, 6500)}

Job description:
{split_text(job_description, 4500)}
"""

    return ask_ollama(
        [{"role": "user", "content": prompt}],
        temperature=0.3
    )


# ============================================================
# SECTION 4: INTERVIEW PREPARATION
# ============================================================

def generate_interview_preparation(
    resume_text,
    job_description,
    job_role
):
    """Create technical, project, and behavioral interview questions."""

    prompt = f"""
You are an interview preparation coach.

Target role: {job_role}

Create a personalized interview preparation guide using the resume
and job description.

Include:

## Technical Questions
Provide 8 questions relevant to the job requirements.

## Project Questions
Provide 5 questions based on projects actually mentioned in the resume.

## Behavioral Questions
Provide 5 questions about teamwork, problem-solving, learning,
communication, and handling challenges.

## Important Topics to Revise
List the most relevant concepts and technologies.

## Preparation Strategy
Provide a short, practical study plan.

Rules:
- Keep the questions appropriate for the target role.
- Do not invent projects or claim the candidate has experience
  that is not supported by the resume.
- Avoid unrelated advanced questions unless the job description requires them.

Resume:
{split_text(resume_text, 6000)}

Job description:
{split_text(job_description, 4000)}
"""

    return ask_ollama(
        [{"role": "user", "content": prompt}],
        temperature=0.4
    )


# ============================================================
# INTERVIEW HELPERS
# ============================================================

def get_interview_history():
    """Return actual interviewer questions and candidate answers."""

    return [
        message
        for message in st.session_state.messages
        if message.get("role") in ("assistant", "user")
    ]


def generate_first_question():
    """Generate the first personalized interview question."""

    prompt = f"""
You are conducting a realistic interview for the role:
{st.session_state.job_role}

Use the resume and job description to personalize the interview.

Ask exactly ONE clear question.
Start with a suitable opening question.
Do not provide the answer or ask multiple questions at once.

Resume:
{split_text(st.session_state.resume_text, 4500)}

Job description:
{split_text(st.session_state.job_description, 3000)}
"""

    return ask_ollama(
        [{"role": "user", "content": prompt}],
        temperature=0.4
    )


def generate_next_question():
    """Generate an adaptive follow-up based on the candidate's answers."""

    history = get_interview_history()

    transcript = "\n".join(
        f"{'Interviewer' if item['role'] == 'assistant' else 'Candidate'}: "
        f"{item['content']}"
        for item in history
    )

    prompt = f"""
You are conducting an adaptive interview for:
{st.session_state.job_role}

Ask exactly ONE next interview question.

Requirements:
- Consider the candidate's previous answers.
- Ask a relevant follow-up if an answer was incomplete or unclear.
- Otherwise, move to another important skill, project, or job requirement.
- Do not repeat a question already asked.
- Use a mix of technical, project-based, and behavioral questions.
- Keep the difficulty suitable for the target role.
- Do not provide the answer.
- Output only the next question.

Resume:
{split_text(st.session_state.resume_text, 3500)}

Job description:
{split_text(st.session_state.job_description, 2500)}

Interview transcript:
{transcript}
"""

    return ask_ollama(
        [{"role": "user", "content": prompt}],
        temperature=0.4
    )


# ============================================================
# SECTION 6: FEEDBACK AND EVALUATION
# ============================================================

def generate_evaluation_report():
    """Evaluate submitted interview answers and create a learning plan."""

    messages = st.session_state.messages

    answers = [
        message["content"].strip()
        for message in messages
        if message.get("role") == "user"
        and message.get("content", "").strip()
    ]

    if not answers:
        return None

    transcript = "\n\n".join(
        (
            "Interviewer: " if message["role"] == "assistant"
            else "Candidate: "
        ) + message["content"]
        for message in messages
    )

    prompt = f"""
You are an interview evaluator.

Evaluate the candidate's submitted answers for the role:
{st.session_state.job_role}

Use the transcript as evidence.

Provide:

## Overall Performance
Give an approximate score out of 100 and explain its limitations.

## Technical Knowledge
Evaluate correctness and understanding based on submitted answers.

## Relevance
Evaluate how directly the answers address the questions.

## Completeness
Evaluate whether the candidate explains their reasoning sufficiently.

## Strengths
List strengths demonstrated by the answers.

## Areas for Improvement
Identify specific weaknesses supported by the transcript.

## Recommended Topics to Revise
Give targeted topics based on gaps in the answers.

## Practice Questions
Provide 5 relevant practice questions.

## Personalized Learning Plan
Provide a practical 7-day revision plan.

IMPORTANT:
- Evaluate only answers actually submitted.
- Do not assume an unanswered question was answered incorrectly.
- Do not fabricate technical mistakes or achievements.
- Clearly label scores as approximate AI-generated feedback, not a
  professional hiring decision.
- If evidence is insufficient for a particular assessment, say so.

Resume:
{split_text(st.session_state.resume_text, 3500)}

Job description:
{split_text(st.session_state.job_description, 2500)}

Interview transcript:
{transcript}
"""

    return ask_ollama(
        [{"role": "user", "content": prompt}],
        temperature=0.2
    )


# ============================================================
# DOWNLOAD HELPER
# ============================================================

def download_text_button(label, content, filename):
    """Provide a downloadable text report."""

    st.download_button(
        label=label,
        data=content,
        file_name=filename,
        mime="text/plain",
        key=f"download_{filename}"
    )


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("AI Resume & Interview Assistant")

st.caption("Personalized Career Preparation Using Generative AI")


# ============================================================
# SECTION 1: INPUT
# ============================================================

st.header("1. Input: Resume and Job Description")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF)",
        type=["pdf"],
        key="resume_upload"
    )

with col2:
    job_role = st.text_input(
        "Target job role",
        placeholder="e.g., Python Intern",
        value=st.session_state.job_role
    )

job_description = st.text_area(
    "Paste the full job description",
    height=180,
    value=st.session_state.job_description,
    placeholder="Paste the responsibilities, required skills, qualifications..."
)

analyze_clicked = st.button(
    "Analyze Resume and Prepare Interview",
    type="primary"
)

if analyze_clicked:
    if uploaded_file is None:
        st.error("Please upload your resume PDF.")

    elif not job_role.strip():
        st.error("Please enter the target job role.")

    elif not job_description.strip():
        st.error("Please paste the full job description.")

    else:
        try:
            # Only reset interview state when a new analysis is started.
            st.session_state.job_role = job_role.strip()
            st.session_state.job_description = job_description.strip()

            st.session_state.match_result = ""
            st.session_state.similarity_score = None
            st.session_state.improvement_report = ""
            st.session_state.prep_report = ""
            st.session_state.messages = []
            st.session_state.interview_started = False
            st.session_state.interview_finished = False
            st.session_state.evaluation_report = ""
            st.session_state.interview_error = ""

            with st.spinner("Extracting resume text..."):
                resume_text = extract_resume_text(uploaded_file)

            st.session_state.resume_text = resume_text

            st.success(
                f"Resume extracted successfully "
                f"({len(resume_text):,} characters)."
            )

            with st.spinner(
                "Analyzing resume and job description with local AI..."
            ):
                result = compare_resume_to_jd(
                    resume_text,
                    st.session_state.job_description
                )

                st.session_state.match_result = result["report"]
                st.session_state.similarity_score = result["similarity"]

            st.success("Resume and job description analysis completed.")

        except Exception as error:
            st.error(
                f"Processing failed: {error}\n\n"
                "Check that Ollama is running and the required models "
                "are installed."
            )


# ============================================================
# SECTION 2: RESUME AND JD ANALYSIS
# ============================================================

if st.session_state.match_result:
    st.divider()
    st.header("2. Resume and Job Description Analysis")

    if st.session_state.similarity_score is not None:
        st.metric(
            "Semantic Similarity",
            f"{st.session_state.similarity_score}%"
        )

        st.caption(
            "This is semantic text similarity only, not a validated "
            "ATS score or a probability of getting hired."
        )

    st.markdown(st.session_state.match_result)


# ============================================================
# SECTION 3: RESUME IMPROVEMENT
# ============================================================

if st.session_state.resume_text:
    st.divider()
    st.header("3. Resume Improvement")

    st.write(
        "Generate role-specific suggestions and truthful rewrites "
        "of existing resume content."
    )

    if st.button("Generate Resume Improvement Suggestions"):
        try:
            with st.spinner("Generating resume improvement suggestions..."):
                report = generate_resume_improvements(
                    st.session_state.resume_text,
                    st.session_state.job_description,
                    st.session_state.job_role
                )

            st.session_state.improvement_report = report

        except Exception as error:
            st.error(f"Could not generate suggestions: {error}")

    if st.session_state.improvement_report:
        st.subheader("Practical Resume Improvement Report")
        st.markdown(st.session_state.improvement_report)

        download_text_button(
            "Download Resume Improvement Report",
            st.session_state.improvement_report,
            "resume_improvement_report.txt"
        )


# ============================================================
# SECTION 4: INTERVIEW PREPARATION
# ============================================================

if st.session_state.resume_text:
    st.divider()
    st.header("4. Interview Preparation")

    st.write(
        "Generate technical, project-based, and behavioral questions "
        "personalized to your resume and target job."
    )

    if st.button("Generate Interview Preparation Questions"):
        try:
            with st.spinner("Preparing personalized interview questions..."):
                report = generate_interview_preparation(
                    st.session_state.resume_text,
                    st.session_state.job_description,
                    st.session_state.job_role
                )

            st.session_state.prep_report = report

        except Exception as error:
            st.error(f"Could not generate interview questions: {error}")

    if st.session_state.prep_report:
        st.markdown(st.session_state.prep_report)

        download_text_button(
            "Download Interview Preparation Questions",
            st.session_state.prep_report,
            "interview_preparation.txt"
        )


# ============================================================
# SECTION 5: PERSONALIZED INTERVIEW CHATBOT
# ============================================================

if st.session_state.resume_text:
    st.divider()
    st.header("5. Personalized Interview Chatbot")

    st.caption(f"Target role: {st.session_state.job_role}")

    if (
        not st.session_state.interview_started
        and not st.session_state.interview_finished
    ):
        if st.button("Start Interview", type="primary"):
            try:
                with st.spinner("Preparing your first interview question..."):
                    first_question = generate_first_question()

                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": first_question
                    }
                ]

                st.session_state.interview_started = True
                st.session_state.interview_error = ""
                st.rerun()

            except Exception as error:
                st.error(f"Could not start the interview: {error}")

    # Display the interview transcript.
    if st.session_state.messages:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Keep the answer input and End Interview button together.
    # End Interview appears immediately below the input.
    if (
        st.session_state.interview_started
        and not st.session_state.interview_finished
    ):
        with st.container():
            answer = st.chat_input(
                "Type your answer here...",
                key="interview_answer_input"
            )

            if answer:
                # Store the submitted answer.
                st.session_state.messages.append({
                    "role": "user",
                    "content": answer.strip()
                })

                st.session_state.interview_error = ""

                # Generate the next adaptive question.
                try:
                    with st.spinner("Evaluating your response..."):
                        next_question = generate_next_question()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": next_question
                    })

                except Exception as error:
                    st.session_state.interview_error = str(error)

                # Refresh the transcript and input.
                st.rerun()

            if st.button(
                "End Interview",
                type="secondary",
                key="end_interview_button"
            ):
                st.session_state.interview_finished = True
                st.session_state.interview_started = False
                st.rerun()

        if st.session_state.interview_error:
            st.warning(
                "Your answer was saved, but the next question could not "
                "be generated. Check Ollama and try submitting another "
                "answer.\n\n"
                f"Details: {st.session_state.interview_error}"
            )

    if st.session_state.interview_finished:
        st.success("Interview ended.")

        if st.button(
            "Generate Final Evaluation",
            type="primary",
            key="generate_evaluation_button"
        ):
            try:
                with st.spinner("Evaluating your interview answers..."):
                    report = generate_evaluation_report()

                if report:
                    st.session_state.evaluation_report = report
                else:
                    st.warning(
                        "No answers were submitted. Start a new interview "
                        "and answer some questions before requesting an "
                        "evaluation."
                    )

            except Exception as error:
                st.error(f"Evaluation failed: {error}")


# ============================================================
# SECTION 6: FEEDBACK AND OUTPUT
# ============================================================

if st.session_state.resume_text:
    st.divider()
    st.header("6. Feedback and Output")

    if st.session_state.evaluation_report:
        st.markdown(st.session_state.evaluation_report)

        download_text_button(
            "Download Final Evaluation Report",
            st.session_state.evaluation_report,
            "final_interview_evaluation.txt"
        )

    elif st.session_state.interview_finished:
        submitted_answers = [
            message for message in st.session_state.messages
            if message.get("role") == "user"
            and message.get("content", "").strip()
        ]

        if submitted_answers:
            st.info(
                "Your interview has ended. Select "
                "'Generate Final Evaluation' in Section 5 to view "
                "your personalized feedback."
            )
        else:
            st.subheader("Insufficient Interview Data")
            st.write(
                "No answers were submitted. Start a new interview and "
                "answer some questions before requesting an evaluation."
            )


# ============================================================
# SECTION 7: CONTINUOUS FEEDBACK AND IMPROVEMENT
# ============================================================

if st.session_state.resume_text:
    st.divider()
    st.header("7. Continuous Feedback and Improvement")

    st.write(
        "Use your evaluation report to identify weak topics, revise them, "
        "and practice again. This version does not automatically track "
        "long-term progress between sessions."
    )

    if st.button("Start a New Interview", key="new_interview_button"):
        st.session_state.messages = []
        st.session_state.interview_started = False
        st.session_state.interview_finished = False
        st.session_state.evaluation_report = ""
        st.session_state.interview_error = ""

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Local AI: Ollama + Llama 3.2 + Nomic Embeddings. "
    "AI-generated analyses and scores may be imperfect. "
    "Verify suggestions before using them in a real resume."
)
