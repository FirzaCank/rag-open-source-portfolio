# LinkedIn post

Three versions. Post the long one, keep the short one for a comment reply or a repost later.
No number in here is unmeasured. If a figure changes before you post, change it here first.

Post it after the deck exists, and attach the deck as a PDF carousel. LinkedIn renders a document
carousel inline, which gets read far more than a link.

---

## Long version, the one to post

I am a Data Engineer. My job is getting data correct, on time, and in the right shape. The layer
above that is where most of the value is being created right now, and I did not want to only consume
it.

So I spent the last few months in Rubythalib AI Super Class, working through machine learning, NLP,
computer vision, LLMs, and deployment.

For the final project I picked RAG. Not because it is the trendy choice, but because it is the one
topic on that list that is mostly a data problem. The model is interchangeable. The corpus, the
chunking, and the ranking are not. That is my existing job pointed at a new target.

What I built: the chat assistant on my portfolio site, rebuilt on fully open source models. Qwen for
generation, multilingual e5 for embeddings, numpy for vector search, FastAPI for serving, running on
a Cloud Run GPU that scales to zero so it costs nothing while nobody is asking.

Three things I did not expect to learn.

1. Most of my checks were lying to me. My first baseline scored 28 out of 41 test cases. I ran the
exact same configuration again and got 23. A five point spread means any improvement worth less than
five points is unmeasurable, and I nearly published two of those. Fixing the sampling seed brought it
to 41 out of 41 byte identical answers, and every experiment after that point is seeded.

2. A rule in the system prompt is a request, not a control. The prompt forbade revealing the internal
tool names from the very first version. They leaked in 8 measured runs out of 8, in three languages,
under two different prompts. The reason is simple once you see it: the model receives every tool
schema in the request body each time, and no instruction takes back what your own request format
already handed over. The fix is deterministic code in one place, not a stronger sentence.

3. The highest score is not automatically the winner. A newer model scored the best result the project
has recorded. I refused to promote it, because in that same run it called the contact tool with an
email address the visitor never typed. It invented a person and acted on their behalf. A pass rate
does not price that, and hand classifying every failure is the only reason I saw it at all.

The code is public, the decisions are written down with the evidence that produced them, and the
failures are in there too.

Repo: github.com/FirzaCank/rag-open-source-portfolio

#DataEngineering #RAG #LLM #MLOps #OpenSource #GoogleCloud #AI

---

## Short version, for a comment or a repost

I rebuilt the chat assistant on my portfolio site on fully open source models: Qwen, multilingual e5,
numpy for vector search, FastAPI, on a Cloud Run GPU that scales to zero.

The interesting part was not that it works. It was that my first baseline scored 28 out of 41, and
the identical configuration then scored 23. Everything I thought I had improved before I fixed the
sampling seed was noise.

Repo in the comments.

---

## One line, for the deck's closing slide or a Twitter crosspost

Built a RAG assistant on fully open source models, then spent most of the project discovering that my
own checks were lying to me.

---

## Notes on posting

- Attach the deck as a PDF, not as a link. A document carousel is read inline
- The repo link goes in the post body, not the first comment. LinkedIn no longer penalises it enough to matter, and a link in the body survives being reposted
- Do not add the live assistant URL until the site is deployed and you have watched a cold start finish. First impression on a 92 second cold start with no explanation is a broken product
- Reply to every comment in the first two hours. That is the whole distribution mechanism
