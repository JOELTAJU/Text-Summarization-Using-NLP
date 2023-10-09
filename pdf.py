import newspaper
from newspaper import Article
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from textwrap import fill
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def extract_summary(url, max_sentences=5):
    # Initialize newspaper and Article
    article = Article(url)
    article.download()
    article.parse()

    # Extract full text
    full_text = article.text

    # Generate summary using LSA summarizer from sumy library
    parser = PlaintextParser.from_string(full_text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary_sentences = summarizer(parser.document, max_sentences)

    # Format summary sentences
    summary = " ".join([str(sentence) for sentence in summary_sentences])

    return summary, full_text

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle

# ...

def save_as_pdf(summary, full_text, filename):
    # Create SimpleDocTemplate object
    doc = SimpleDocTemplate(filename, pagesize=letter)

    # Create stylesheet
    styles = getSampleStyleSheet()

    # Create story elements
    story = []

    # Add full text section
    story.append(Paragraph("<b>Full Text:</b>", styles["Heading1"]))
    story.append(Spacer(1, 12))
    for line in full_text.split("\n"):
        story.append(Paragraph(line, styles["Normal"]))
    story.append(Spacer(1, 12))

    # Add summary section
    summary_style = ParagraphStyle("SummaryStyle", parent=styles["Normal"], alignment=TA_CENTER)
    story.append(Paragraph("<b>Summary:</b>", styles["Heading1"]))
    story.append(Spacer(1, 12))
    formatted_summary = fill(summary, width=80)  # Adjust the line width as needed
    for line in formatted_summary.split("\n"):
        paragraph = Paragraph(line, summary_style)
        story.append(paragraph)
        story.append(Spacer(1, 12))

    # Build the PDF document
    doc.build(story)

# ...


# Example usage
url = "https://www.washingtonpost.com/outlook/tale-of-a-woman-who-died-and-a-woman-who-killed-in-the-northern-ireland-conflict/2019/03/08/59e75dd4-2ecd-11e9-8ad3-9a5b113ecd3c_story.html"
summary, full_text = extract_summary(url, max_sentences=10)  # Specify the desired summary length in sentences
save_as_pdf(summary, full_text, "summary.pdf")
