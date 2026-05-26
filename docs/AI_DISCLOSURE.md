# AI Use Disclosure

If you are reading this, you probably followed a link in my resume or cover letter.
This link indicates that I used AI to assist in writing the letter or aligning the resume.

## How AI was used

A master resume containing the my full employment history, skills, projects, and education serves as the sole source of truth. For each application, an LLM reads that master resume alongside the job description and produces a tailored selection of content: which experience bullets to include, which skills are relevant, whether projects add signal, and how to phrase the cover letter body.

The LLM does not invent or add information. Every claim in the resume and cover letter is grounded in the master resume. Formatting, including section structure, contact headers, dates, and layout, is handled by deterministic Python code, not the LLM.

The cover letter body paragraphs are generated from the same master resume with instructions to draw only from documented experience and to write in my voice.

I am the developer of all code used for these materials.

## Human review

I review the application before submission to verify factual accuracy, remove or correct anything that does not reflect my experience, and confirm that the final documents represents me honestly.
