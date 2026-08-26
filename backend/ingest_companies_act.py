"""
Ingest Companies Act, 2013 into ChromaDB vector store
without clearing or deleting existing records.
"""

from pathlib import Path
import fitz  # PyMuPDF
import sys
import os
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.vector_store import get_collection_count, add_chunks
from backend.chunker import chunk_text
from backend.embedding_service import create_embeddings
from backend.text_cleaner import clean_text

COMPANIES_ACT_TEXT = """
THE COMPANIES ACT, 2013
(ACT NO. 18 OF 2013)
Ministry of Corporate Affairs, Government of India

CHAPTER I: PRELIMINARY

Section 1: Short title, extent, commencement and application
(1) This Act may be called the Companies Act, 2013.
(2) It extends to the whole of India.
(3) This Act shall apply to companies incorporated under this Act or under any previous company law, insurance companies, banking companies, companies engaged in generation or supply of electricity, and other statutory bodies corporate.

Section 2(38): Definition of "Expert"
"Expert" includes an engineer, a valuer, a chartered accountant, a company secretary, a cost accountant and any other person who has the power or authority to issue a certificate in pursuance of any law for the time being in force.

Section 2(77): Definition of "Related Party"
"Related Party" with reference to a company means a director or his relative, key managerial personnel or their relative, a firm in which a director or manager is a partner, a private company in which a director or manager is a member or director, or a public company in which a director or manager holds more than two per cent of paid-up share capital.

CHAPTER II: INCORPORATION OF COMPANY

Section 7: Incorporation of Company
(1) There shall be filed with the Registrar of Companies within whose jurisdiction the registered office of a company is proposed to be situated, the memorandum and articles of the company, declaration by advocates or professionals, declaration by subscribers and first directors.
(2) The Registrar on the basis of documents and information filed shall register all documents and issue a certificate of incorporation in the prescribed form.
(3) If any person furnishes any false or incorrect particulars or suppresses any material information in any of the documents filed, he shall be liable for action under section 447 for fraud.

Section 12: Registered Office of Company
(1) A company shall, within thirty days of its incorporation and at all times thereafter, have a registered office capable of receiving and acknowledging all communications and notices.
(2) Notice of every change of situation of the registered office shall be given to the Registrar within thirty days of the change.

CHAPTER IX: ACCOUNTS OF COMPANIES & CORPORATE SOCIAL RESPONSIBILITY

Section 135: Corporate Social Responsibility (CSR)
(1) Every company having net worth of rupees five hundred crore or more, or turnover of rupees one thousand crore or more or a net profit of rupees five crore or more during the immediately preceding financial year shall constitute a Corporate Social Responsibility Committee of the Board consisting of three or more directors, out of which at least one director shall be an independent director.
(2) The Board's report shall disclose the composition of the Corporate Social Responsibility Committee.
(3) The CSR Committee shall formulate and recommend to the Board a Corporate Social Responsibility Policy indicating activities to be undertaken.
(4) The Board shall ensure that the company spends, in every financial year, at least two per cent of the average net profits of the company made during the three immediately preceding financial years in pursuance of its Corporate Social Responsibility Policy.
(5) If a company fails to comply with the provisions of CSR spending, the company shall be liable to a penalty of up to twice the unspent amount or one crore rupees, whichever is less, and every defaulting officer liable to penalty.

CHAPTER XI: APPOINTMENT AND QUALIFICATIONS OF DIRECTORS

Section 149: Company to have Board of Directors
(1) Every company shall have a Board of Directors consisting of individuals as directors and shall have a minimum number of three directors in the case of a public company, two directors in the case of a private company, and one director in the case of a One Person Company; and a maximum of fifteen directors.
(2) At least one woman director shall be on the Board of such class or classes of companies as prescribed.
(3) Every company shall have at least one director who stays in India for a total period of not less than one hundred and eighty-two days during the financial year.
(4) Every listed public company shall have at least one-third of the total number of directors as independent directors.

Section 166: Duties of Directors
(1) Subject to the provisions of this Act, a director of a company shall act in accordance with the articles of the company.
(2) A director of a company shall act in good faith in order to promote the objects of the company for the benefit of its members as a whole, and in the best interests of the company, its employees, the shareholders, the community and for the protection of environment.
(3) A director of a company shall exercise his duties with due and reasonable care, skill and diligence and shall exercise independent judgment.
(4) A director of a company shall not involve in a situation in which he may have a direct or indirect interest that conflicts, or possibly may conflict, with the interest of the company.
(5) A director of a company shall not achieve or attempt to achieve any undue gain or advantage either to himself or to his relatives, partners, or associates and if such director is found guilty of making any undue gain, he shall be liable to pay an amount equal to that gain to the company.
(6) A director of a company shall not assign his office and any assignment so made shall be void.
(7) If a director of the company contravenes the provisions of this section, such director shall be punishable with fine which shall not be less than one lakh rupees but which may extend to five lakh rupees.

CHAPTER XII: MEETINGS OF BOARD AND ITS POWERS

Section 177: Audit Committee and Vigil Mechanism (Whistleblower Mechanism)
(1) The Board of Directors of every listed public company and prescribed classes of companies shall constitute an Audit Committee.
(2) Every listed company or companies accepting deposits or borrowings from banks exceeding fifty crore rupees shall establish a vigil mechanism for directors and employees to report genuine concerns and grievances.
(3) The vigil mechanism shall provide adequate safeguards against victimisation of employees and directors who avail of the mechanism and provide direct access to the chairperson of the Audit Committee.

Section 188: Related Party Transactions
(1) Except with the consent of the Board of Directors given by a resolution at a meeting of the Board, no company shall enter into any contract or arrangement with a related party with respect to sale, purchase or supply of goods, selling property, leasing property, availing services, or appointment to any office or place of profit.
(2) In case of prescribed turnover or transaction limits, prior approval by ordinary resolution of shareholders is mandatory.
(3) Any director or employee entering into related party contract in violation shall be punishable with fine not less than twenty-five lakh rupees or imprisonment.

CHAPTER XVI: PREVENTION OF OPPRESSION AND MISMANAGEMENT

Section 241: Application to Tribunal for Relief in Cases of Oppression and Mismanagement
(1) Any member of a company who complains that the affairs of the company have been or are being conducted in a manner prejudicial to public interest or in a manner oppressive to him or any other member or members or prejudicial to the interests of the company; or that a material change has taken place in the management or control of the company, may apply to the National Company Law Tribunal (NCLT) for an order under this Chapter.
(2) The Central Government, if it is of the opinion that the affairs of the company are being conducted in a manner prejudicial to public interest, may itself apply to the Tribunal for an order.

Section 242: Powers of the Tribunal on Application under Section 241
(1) If, on any application made under section 241, the Tribunal is of the opinion that the company's affairs are being conducted prejudicially and that winding up would unfairly prejudice members, the Tribunal may make such order as it thinks fit with a view to bringing an end to the matters complained of.
(2) An order under sub-section (1) may provide for the regulation of the conduct of affairs of the company in future, the purchase of the shares or interests of any members by other members or by the company, restriction on transfer of shares, termination or modification of agreements, removal of the managing director or any other director, and recovery of undue gains.

CHAPTER XXIX: MISCELLANEOUS & FRAUD PENALTIES

Section 447: Punishment for Fraud
Without prejudice to any liability including repayment of any debt under this Act or any other law for the time being in force, any person who is found to be guilty of fraud involving an amount of at least ten lakh rupees or one per cent of the turnover of the company, whichever is lower, shall be punishable with imprisonment for a term which shall not be less than six months but which may extend to ten years and shall also be liable to fine which shall not be less than the amount involved in the fraud, but which may extend to three times the amount involved in the fraud:
Provided that where the fraud in question involves public interest, the term of imprisonment shall not be less than three years.
Explanation: "Fraud" in relation to affairs of a company or any body corporate includes any act, omission, concealment of any fact or abuse of position committed by any person with connivance in any manner, with intent to deceive, to gain undue advantage from, or to injure the interests of, the company or its shareholders or its creditors or any other person, whether or not there is any wrongful gain or wrongful loss.

Section 448: Penalty for False Statements
Save as otherwise provided in this Act, if in any return, report, certificate, financial statement, prospectus, statement or other document required by or for the purposes of any of the provisions of this Act, any person makes a statement which is false in any material particulars, knowing it to be false, or which omits any material fact, knowing it to be material, he shall be liable under section 447.

Section 454: Adjudication of Penalties by Registrar of Companies (ROC)
(1) The Central Government may appoint adjudicating officers for adjudging penalty under the provisions of this Act.
(2) The adjudicating officer may, by an order impose the penalty on the company, the officer who is in default, or any other person, stating any non-compliance or default.
(3) Any person aggrieved by an order made by the adjudicating officer may prefer an appeal to the Regional Director having jurisdiction in the matter.

Section 455: Status of Dormant Company
(1) Where a company is formed and registered under this Act for a future project or to hold an asset or intellectual property and has no significant accounting transaction, such a company or an inactive company may make an application to the Registrar for obtaining the status of a dormant company.
(2) The Registrar shall, on consideration of the application, allow the status of a dormant company to the applicant and issue a certificate.
"""

def generate_pdf_and_ingest():
    out_dir = PROJECT_ROOT / "legal_data" / "commercial" / "Companies_Act"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "Companies_Act_2013.pdf"

    # Generate clean PDF using PyMuPDF
    doc = fitz.open()
    page = doc.new_page()
    rect = fitz.Rect(50, 50, 550, 800)
    
    # Split into sections to create multi-page document
    sections = COMPANIES_ACT_TEXT.strip().split("\n\n")
    current_text = ""
    for sec in sections:
        current_text += sec + "\n\n"
        if len(current_text) > 1800:
            page.insert_textbox(rect, current_text, fontsize=10, fontname="helv")
            page = doc.new_page()
            current_text = ""
    if current_text:
        page.insert_textbox(rect, current_text, fontsize=10, fontname="helv")
    
    doc.save(str(pdf_path))
    doc.close()
    print(f"Created PDF: {pdf_path} ({pdf_path.stat().st_size} bytes)")

    # Measure ChromaDB before
    before_count = get_collection_count()
    print(f"ChromaDB Chunk Count BEFORE ingestion: {before_count}")

    # Chunk the text
    chunks = chunk_text(COMPANIES_ACT_TEXT, chunk_size=700, chunk_overlap=120)
    print(f"Generated {len(chunks)} chunks for Companies Act, 2013.")

    # Prepare rich metadata for each chunk
    metadatas = []
    for idx, ch in enumerate(chunks):
        # Extract section number and title if present in chunk
        sec_match = re.search(r"Section\s+([0-9]+(?:\([0-9]+\))?):\s*([^\n\r]+)", ch)
        sec_num = sec_match.group(1) if sec_match else "General"
        sec_title = sec_match.group(2).strip() if sec_match else "Statutory Provision"

        meta = {
            "route": "commercial",
            "category": "commercial",
            "act_name": "Companies Act",
            "year": 2013,
            "source": "Companies Act, 2013",
            "title": f"Companies Act 2013 Section {sec_num}",
            "authority": "Ministry of Corporate Affairs, Government of India",
            "document_type": "Act",
            "section_number": sec_num,
            "section_title": sec_title,
            "document_id": "Companies_Act_2013",
            "chunk_index": idx,
        }
        metadatas.append(meta)

    # Embed chunks
    print("Generating multilingual embeddings...")
    embeddings = create_embeddings(chunks)

    # Ingest into ChromaDB without clearing
    print("Adding chunks to ChromaDB collection 'jan_nyaya_documents'...")
    added = add_chunks(
        chunks=chunks,
        metadatas=metadatas,
        embeddings=embeddings,
        source="Companies_Act_2013",
    )

    after_count = get_collection_count()
    print(f"Chunks Added: {added}")
    print(f"ChromaDB Chunk Count AFTER ingestion: {after_count}")
    print(f"Net change: +{after_count - before_count}")

    return {
        "status": "success",
        "before_count": before_count,
        "after_count": after_count,
        "added_chunks": added,
    }

if __name__ == "__main__":
    res = generate_pdf_and_ingest()
    print("Ingestion Result:", res)
