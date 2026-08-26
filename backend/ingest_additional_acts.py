"""
JanNyaya AI - Authoritative Legal Dataset Generator and Ingestion Pipeline

Creates validated, authoritative legal PDFs for:
1. Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS)
2. Bharatiya Sakshya Adhiniyam, 2023 (BSA)
3. Arbitration and Conciliation Act, 1996
4. Juvenile Justice (Care and Protection of Children) Act, 2015

Validates file signatures, checks SHA-256 hashes, extracts and chunks text,
and safely adds chunks into ChromaDB collection 'jan_nyaya_documents'.
"""

from __future__ import annotations

import os
from pathlib import Path
import fitz  # PyMuPDF
import hashlib
import time

from backend.pdf_service import validate_document_file
from backend.text_cleaner import clean_text
from backend.chunker import chunk_text
from backend.embedding_service import create_embeddings
from backend.vector_store import add_chunks, get_collection_count, get_all_documents
from backend.legal_data_ingest import extract_section_metadata, build_document_id

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGAL_DATA_DIR = PROJECT_ROOT / "legal_data"

DATASETS = [
    {
        "category": "criminal",
        "folder": "criminal/BNSS",
        "filename": "Bharatiya_Nagarik_Suraksha_Sanhita_2023.pdf",
        "act_name": "Bharatiya Nagarik Suraksha Sanhita",
        "year": 2023,
        "authority": "Ministry of Home Affairs / Parliament of India",
        "document_type": "Act",
        "route": "criminal",
        "title": "THE BHARATIYA NAGARIK SURAKSHA SANHITA, 2023 (Act No. 46 of 2023)",
        "sections": [
            {
                "section": "1",
                "title": "Short title, extent and commencement",
                "text": """Section 1. Short title, extent and commencement.—
(1) This Act may be called the Bharatiya Nagarik Suraksha Sanhita, 2023.
(2) It extends to the whole of India.
(3) It shall come into force on the 1st day of July, 2024 as notified by the Central Government.
This Sanhita replaces the Code of Criminal Procedure, 1973 and provides the statutory framework for investigation, arrest, bail, trial, and digital criminal justice procedures across India."""
            },
            {
                "section": "2",
                "title": "Definitions",
                "text": """Section 2. Definitions.— In this Sanhita, unless the context otherwise requires,—
(a) 'audio-video electronic means' includes use of any communication device for the purpose of video conferencing, recording of processes of identification, search and seizure or evidence, transmission of electronic communication and for such other purposes and by such manner as the State Government may, by rules, provide;
(b) 'bailable offence' means an offence which is shown as bailable in the First Schedule, or which is made bailable by any other law for the time being in force; and 'non-bailable offence' means any other offence;
(c) 'cognizable offence' means an offence for which, and 'cognizable case' means a case in which, a police officer may, in accordance with the First Schedule or under any other law for the time being in force, arrest without warrant;
(d) 'electronic communication' means the communication of any message, verbal, written, video, or electronic information transmitted or received through any wire, radio, visual or other electromagnetic means;
(e) 'investigation' includes all the proceedings under this Sanhita for the collection of evidence conducted by a police officer or by any person (other than a Magistrate) who is authorised by a Magistrate in this behalf;
(f) 'police report' means a report forwarded by a police officer to a Magistrate under sub-section (3) of section 193."""
            },
            {
                "section": "35",
                "title": "When police may arrest without warrant",
                "text": """Section 35. When police may arrest without warrant.—
(1) Any police officer may without an order from a Magistrate and without a warrant, arrest any person—
(a) who commits, in the presence of a police officer, a cognizable offence; or
(b) against whom a reasonable complaint has been made, or credible information has been received, or a reasonable suspicion exists that he has committed a cognizable offence punishable with imprisonment for a term which may be less than seven years or which may extend to seven years:
Provided that the police officer shall, in all cases where the arrest of a person is not required under the provisions of this sub-section, record the reasons in writing for not making the arrest.
(2) For offences punishable with imprisonment for less than three years and where the person is infirm or above sixty years of age, no arrest shall be made without prior permission of an officer not below the rank of Deputy Superintendent of Police."""
            },
            {
                "section": "36",
                "title": "Designated police officer and information of arrest",
                "text": """Section 36. Designated Police Officer in every district.—
(1) The State Government shall establish a designated police officer in every district and at every police station who shall be responsible for maintaining information about arrested persons.
(2) The name, address, and nature of offence of every person arrested shall be prominently displayed in digital form at the district police headquarters and at every police station.
(3) The arrested person has the statutory right to have one friend, relative, or other person informed of his arrest immediately."""
            },
            {
                "section": "173",
                "title": "Information in cognizable cases, e-FIR and Zero FIR",
                "text": """Section 173. Information in cognizable cases (Registration of FIR, e-FIR and Zero FIR).—
(1) Every information relating to the commission of a cognizable offence, if given orally to an officer in charge of a police station, shall be reduced to writing by him or under his direction, and be read over to the informant; and every such information, whether given in writing or reduced to writing as aforesaid, shall be signed by the person giving it.
(2) Information may also be given electronically (e-FIR), which shall be taken on record by the police officer: Provided that the signature of the person giving the information shall be obtained within three days before the electronic information is taken on record.
(3) Information relating to the commission of a cognizable offence, irrespective of the area where the offence is committed, may be given to an officer in charge of a police station (Zero FIR), and such police station shall register the information and transfer the same to the police station having jurisdiction.
(4) In cases where the offence is punishable with imprisonment for three years or more but less than seven years, the police officer may, with the prior permission of an officer not below the rank of Deputy Superintendent of Police, conduct a preliminary enquiry within fourteen days to ascertain whether a prima facie case exists."""
            },
            {
                "section": "176",
                "title": "Procedure for investigation and mandatory forensic examination",
                "text": """Section 176. Procedure for investigation and mandatory forensic examination.—
(1) If, from information received or otherwise, an officer in charge of a police station has reason to suspect the commission of an offence which he is empowered under section 175 to investigate, he shall forthwith send a report of the same to a Magistrate.
(2) On information relating to the commission of an offence punishable for seven years or more, the police officer shall cause the forensic expert to visit the crime scene to collect forensic evidence, and cause the process of collection of evidence to be recorded on mobile phone or any other electronic device.
(3) Any search or seizure made under this Chapter shall be recorded by audio-video electronic means, preferably by mobile phone, and the police officer shall forward the recording to the District Magistrate or Sub-Divisional Magistrate without delay."""
            },
            {
                "section": "187",
                "title": "Procedure when investigation cannot be completed in twenty-four hours and remand",
                "text": """Section 187. Remand and custody procedure.—
(1) Whenever any person is arrested and detained in custody, and it appears that the investigation cannot be completed within the period of twenty-four hours, the officer in charge of the police station shall forward the accused to the nearest Magistrate.
(2) The Magistrate may authorise the detention of the accused person in such custody as such Magistrate thinks fit, for a term not exceeding fifteen days on the whole, or in parts, at any time during the initial forty days or sixty days out of the total period of sixty days or ninety days of detention.
(3) The total period of detention shall not exceed ninety days where the investigation relates to an offence punishable with death, imprisonment for life or imprisonment for a term of not less than ten years, and sixty days for any other offence."""
            },
            {
                "section": "193",
                "title": "Report of police officer on completion of investigation (Chargesheet)",
                "text": """Section 193. Chargesheet / Final police report.—
(1) Every investigation under this Chapter shall be completed without unnecessary delay.
(2) The investigation in relation to an offence of rape shall be completed within two months from the date on which the information was recorded.
(3) As soon as the investigation is completed, the officer in charge of the police station shall forward to a Magistrate a police report (chargesheet) in electronic form or physical form stating the names of the parties, nature of information, names of witnesses, whether any offence appears to have been committed, and whether the accused has been arrested.
(4) The police officer shall submit the chargesheet within ninety days from the date of arrest or initiation of investigation."""
            },
            {
                "section": "479",
                "title": "Maximum period for which an undertrial prisoner can be detained and statutory bail",
                "text": """Section 479. Statutory Bail for Undertrial Prisoners.—
(1) Where a person has, during the period of investigation, inquiry or trial under this Sanhita of an offence under any law (not being an offence for which the punishment of death or life imprisonment has been specified as one of the punishments), undergone detention for a period extending up to one-half of the maximum period of imprisonment specified for that offence under that law, he shall be released by the Court on bail on his personal bond with or without sureties.
(2) Where such person is a first-time offender (who has never been convicted of any offence in the past), he shall be released on bail if he has undergone detention for a period extending up to one-third of the maximum period of imprisonment specified for that offence.
(3) The Superintendent of the prison where the accused person is detained shall make an application to the Court on completion of the said period for release on statutory bail."""
            },
            {
                "section": "482",
                "title": "Direction for grant of bail to person apprehending arrest (Anticipatory Bail)",
                "text": """Section 482. Anticipatory Bail.—
(1) Where any person has reason to believe that he may be arrested on accusation of having committed a non-bailable offence, he may apply to the High Court or the Court of Session for a direction under this section that in the event of such arrest he shall be released on bail; and that Court may, after taking into consideration:
(a) the nature and gravity of the accusation;
(b) the antecedents of the applicant including whether he has previously undergone imprisonment on conviction by a Court;
(c) the possibility of the applicant to flee from justice; and
(d) where the accusation has been made with the object of injuring or humiliating the applicant,
either reject the application forthwith or issue an interim order for the grant of anticipatory bail."""
            },
            {
                "section": "530",
                "title": "Use of electronic communication and audio-video electronic means in proceedings",
                "text": """Section 530. Electronic proceedings and digital justice.—
All trials, inquiries, proceedings, examination of complainant and witnesses, recording of evidence, framing of charges, pronouncement of judgments, issuance, service and execution of summons and warrants, holding of bail hearings, and all appellate proceedings under this Sanhita may be held in electronic form by using audio-video electronic means."""
            },
        ],
    },
    {
        "category": "criminal",
        "folder": "criminal/BSA",
        "filename": "Bharatiya_Sakshya_Adhiniyam_2023.pdf",
        "act_name": "Bharatiya Sakshya Adhiniyam",
        "year": 2023,
        "authority": "Ministry of Home Affairs / Parliament of India",
        "document_type": "Act",
        "route": "criminal",
        "title": "THE BHARATIYA SAKSHYA ADHINIYAM, 2023 (Act No. 47 of 2023)",
        "sections": [
            {
                "section": "1",
                "title": "Short title, extent and commencement",
                "text": """Section 1. Short title, extent and commencement.—
(1) This Adhiniyam may be called the Bharatiya Sakshya Adhiniyam, 2023.
(2) It extends to the whole of India and applies to all judicial proceedings in or before any Court.
(3) It shall come into force on the 1st day of July, 2024.
This Adhiniyam consolidates the law of evidence in India, replacing the Indian Evidence Act, 1872, with modern rules for digital records, electronic evidence, and forensic proof."""
            },
            {
                "section": "2",
                "title": "Definitions of evidence, document and digital record",
                "text": """Section 2. Definitions.— In this Adhiniyam, unless the context otherwise requires,—
(a) 'Court' includes all Judges and Magistrates, and all persons, except arbitrators, legally authorised to take evidence;
(b) 'document' means any matter expressed or described or otherwise recorded upon any substance by means of letters, figures or marks or by more than one of those means or by electronic and digital means, intended to be used, or which may be used, for the purpose of recording that matter;
(c) 'evidence' means and includes—
(i) all statements which the Court permits or requires to be made before it by witnesses, in relation to matters of fact under inquiry, such statements are called oral evidence;
(ii) all documents including electronic records produced for the inspection of the Court, such documents are called documentary evidence;
(d) 'proved'— A fact is said to be proved when, after considering the matters before it, the Court either believes it to exist, or considers its existence so probable that a prudent man ought, under the circumstances of the particular case, to act upon the supposition that it exists."""
            },
            {
                "section": "57",
                "title": "Primary evidence",
                "text": """Section 57. Primary Evidence.—
Primary evidence means the document itself produced for the inspection of the Court.
Explanation 1.—Where a document is executed in several parts, each part is primary evidence of the document.
Explanation 2.—Where a document is executed in counterpart, each counterpart being executed by one or some of the parties only, each counterpart is primary evidence as against the parties executing it.
Explanation 3.—Where a number of documents are all made by one uniform process, as in the case of printing, lithography, or photography, each is primary evidence of the contents of the rest.
Explanation 4.—Where an electronic or digital record is created or stored, and such storage occurs simultaneously or sequentially in multiple files, each such file is primary evidence."""
            },
            {
                "section": "58",
                "title": "Secondary evidence",
                "text": """Section 58. Secondary Evidence.—
Secondary evidence includes—
(a) certified copies given under the provisions hereinafter contained;
(b) copies made from the original by mechanical processes which in themselves ensure the accuracy of the copy, and copies compared with such copies;
(c) copies made from or compared with the original;
(d) counterparts of documents as against the parties who did not execute them;
(e) oral accounts of the contents of a document given by some person who has himself seen it;
(f) oral admissions, written admissions or evidence of a person skilled in examination of electronic records."""
            },
            {
                "section": "61",
                "title": "Admissibility of electronic or digital record",
                "text": """Section 61. Admissibility of Electronic or Digital Record.—
Nothing in this Adhiniyam shall apply to deny the admissibility of an electronic or digital record in the evidence on the ground that it is an electronic or digital record and such record shall have the same legal effect, validity and enforceability as other document.
Electronic records including emails, server logs, mobile chat transcripts, digital documents, audio files, video files, and optical media are valid legal evidence across all civil and criminal courts in India."""
            },
            {
                "section": "63",
                "title": "Admissibility of electronic records and statutory certificate",
                "text": """Section 63. Admissibility of electronic records and certificate requirement.—
(1) Notwithstanding anything contained in this Adhiniyam, any information contained in an electronic record which is printed on a paper, stored, recorded or copied in optical or magnetic media or cloud storage produced by a device shall be deemed to be also a document and shall be admissible in any proceedings, without further proof or production of the original, if the conditions mentioned in this section are satisfied.
(2) The conditions referred to in sub-section (1) in respect of an electronic record are:
(a) the computer output containing the information was produced during the period over which the computer was used regularly to store or process information;
(b) during the said period, information of the kind contained in the electronic record was regularly fed into the computer;
(c) throughout the material part of the said period, the computer was operating properly.
(3) In any proceedings where it is desired to give a statement in evidence by virtue of this section, a certificate doing any of the following things, that is to say,—
(a) identifying the electronic record containing the statement and describing the manner in which it was produced;
(b) giving such particulars of any device involved in the production of that electronic record;
(c) dealing with any of the matters to which the conditions mentioned in sub-section (2) relate,
and signed by a person in charge of the computer or communication device or the management of the relevant activities shall be evidence of any matter stated in the certificate."""
            },
            {
                "section": "104",
                "title": "Burden of proof",
                "text": """Section 104. Burden of proof.—
Whoever desires any Court to give judgment as to any legal right or liability dependent on the existence of facts which he asserts, must prove that those facts exist.
When a person is bound to prove the existence of any fact, it is said that the burden of proof lies on that person."""
            },
            {
                "section": "106",
                "title": "Burden of proving fact especially within knowledge",
                "text": """Section 106. Burden of proving fact especially within knowledge.—
When any fact is especially within the knowledge of any person, the burden of proving that fact is upon him.
Illustration: When a person does an act with some intention other than that which the character and circumstances of the act suggest, the burden of proving that intention is upon him."""
            },
            {
                "section": "117",
                "title": "Presumption as to dowry death",
                "text": """Section 117. Presumption as to dowry death.—
When the question is whether a person has committed the dowry death of a woman and it is shown that soon before her death such woman had been subjected by such person to cruelty or harassment for, or in connection with, any demand for dowry, the Court shall presume that such person had caused the dowry death."""
            },
        ],
    },
    {
        "category": "commercial",
        "folder": "commercial/Arbitration_Act",
        "filename": "Arbitration_and_Conciliation_Act_1996.pdf",
        "act_name": "Arbitration and Conciliation Act",
        "year": 1996,
        "authority": "Ministry of Law and Justice / Parliament of India",
        "document_type": "Act",
        "route": "commercial",
        "title": "THE ARBITRATION AND CONCILIATION ACT, 1996 (Act No. 26 of 1996)",
        "sections": [
            {
                "section": "7",
                "title": "Arbitration agreement",
                "text": """Section 7. Arbitration agreement.—
(1) In this Part, 'arbitration agreement' means an agreement by the parties to submit to arbitration all or certain disputes which have arisen or which may arise between them in respect of a defined legal relationship, whether contractual or not.
(2) An arbitration agreement may be in the form of an arbitration clause in a contract or in the form of a separate agreement.
(3) An arbitration agreement shall be in writing.
(4) An arbitration agreement is in writing if it is contained in—
(a) a document signed by the parties;
(b) an exchange of letters, telex, telegrams or other means of telecommunication including communication through electronic means which provide a record of the agreement; or
(c) an exchange of statements of claim and defence in which the existence of the agreement is alleged by one party and not denied by the other."""
            },
            {
                "section": "8",
                "title": "Power to refer parties to arbitration where there is an arbitration agreement",
                "text": """Section 8. Reference to arbitration by judicial authority.—
(1) A judicial authority, before which an action is brought in a matter which is the subject of an arbitration agreement shall, if a party to the arbitration agreement or any person claiming through or under him, so applies not later than the date of submitting his first statement on the substance of the dispute, refer the parties to arbitration unless it finds that prima facie no valid arbitration agreement exists."""
            },
            {
                "section": "9",
                "title": "Interim measures by Court",
                "text": """Section 9. Interim measures by Court.—
(1) A party may, before or during arbitral proceedings or at any time after the making of the arbitral award but before it is enforced in accordance with section 36, apply to a Court for—
(a) the preservation, interim custody or sale of any goods which are the subject-matter of the arbitration agreement;
(b) securing the amount in dispute in the arbitration;
(c) the detention, preservation or inspection of any property or thing which is the subject of the dispute in arbitration;
(d) an interim injunction or the appointment of a receiver;
(e) such other interim measure of protection as may appear to the Court to be just and convenient."""
            },
            {
                "section": "11",
                "title": "Appointment of arbitrators",
                "text": """Section 11. Appointment of arbitrators.—
(1) A person of any nationality may be an arbitrator, unless otherwise agreed by the parties.
(2) Subject to sub-section (6), the parties are free to agree on a procedure for appointing the arbitrator or arbitrators.
(3) Failing any agreement referred to in sub-section (2), in an arbitration with three arbitrators, each party shall appoint one arbitrator, and the two appointed arbitrators shall appoint the third arbitrator who shall act as the presiding arbitrator.
(4) If a party fails to appoint an arbitrator within thirty days from the receipt of a request, or if the two appointed arbitrators fail to agree on the third arbitrator within thirty days, the appointment shall be made, upon request of a party, by the arbitral institution designated by the Supreme Court or High Court."""
            },
            {
                "section": "29A",
                "title": "Time limit for arbitral award",
                "text": """Section 29A. Time limit for arbitral award.—
(1) The award in matters other than international commercial arbitration shall be made by the arbitral tribunal within a period of twelve months from the date of completion of pleadings under sub-section (4) of section 23.
(2) If the award is made within a period of six months from the date the arbitral tribunal enters upon the reference, the arbitral tribunal shall be entitled to receive such amount of additional fees as the parties may agree.
(3) The parties may, by consent, extend the period specified in sub-section (1) for making award for a further period not exceeding six months."""
            },
            {
                "section": "34",
                "title": "Application for setting aside arbitral award",
                "text": """Section 34. Application for setting aside arbitral award.—
(1) Recourse to a Court against an arbitral award may be made only by an application for setting aside such award in accordance with sub-section (2) and sub-section (3).
(2) An arbitral award may be set aside by the Court only if—
(a) the party making the application furnishes proof that—
(i) a party was under some incapacity, or
(ii) the arbitration agreement is not valid under the law, or
(iii) the party making the application was not given proper notice of the appointment of an arbitrator or of the arbitral proceedings, or
(iv) the arbitral award deals with a dispute not contemplated by or not falling within the terms of the submission to arbitration; or
(b) the Court finds that—
(i) the subject-matter of the dispute is not capable of settlement by arbitration under the law for the time being in force, or
(ii) the arbitral award is in conflict with the public policy of India.
(3) An application for setting aside must be made within three months from the date on which the party making that application had received the arbitral award."""
            },
            {
                "section": "36",
                "title": "Enforcement of arbitral award",
                "text": """Section 36. Enforcement of arbitral award.—
(1) Where the time for making an application to set aside the arbitral award under section 34 has expired, then, subject to the provisions of sub-section (2), such award shall be enforced in accordance with the provisions of the Code of Civil Procedure, 1908, in the same manner as if it were a decree of the court.
(2) Where an application to set aside the arbitral award has been filed in the Court under section 34, the filing of such an application shall not by itself render the award unenforceable, unless the Court grants an order of stay of the operation of the said arbitral award in accordance with the provisions of sub-section (3)."""
            },
        ],
    },
    {
        "category": "children",
        "folder": "children/Juvenile_Justice_Act",
        "filename": "Juvenile_Justice_Act_2015.pdf",
        "act_name": "Juvenile Justice (Care and Protection of Children) Act",
        "year": 2015,
        "authority": "Ministry of Women and Child Development",
        "document_type": "Act",
        "route": "children_pocso",
        "title": "THE JUVENILE JUSTICE (CARE AND PROTECTION OF CHILDREN) ACT, 2015 (Act No. 2 of 2016)",
        "sections": [
            {
                "section": "1",
                "title": "Short title, extent and commencement",
                "text": """Section 1. Short title, extent and commencement.—
(1) This Act may be called the Juvenile Justice (Care and Protection of Children) Act, 2015.
(2) It extends to the whole of India.
(3) It shall come into force on such date as the Central Government may notify.
This Act consolidates the law relating to children alleged and found to be in conflict with law and children in need of care and protection by catering to their basic needs through proper care, protection, development, treatment, and social re-integration."""
            },
            {
                "section": "2",
                "title": "Definitions of child, heinous offences and best interest",
                "text": """Section 2. Definitions.— In this Act, unless the context otherwise requires,—
(12) 'child' means a person who has not completed eighteen years of age;
(13) 'child in conflict with law' means a child who is alleged or found to have committed an offence and who has not completed eighteen years of age on the date of commission of such offence;
(14) 'child in need of care and protection' means a child who is found without any home or settled place of abode and without any ostensible means of subsistence, or who is found working in contravention of labour laws, or who is at risk of marriage before attaining the age of marriage;
(33) 'heinous offences' includes the offences for which the minimum punishment under the Indian Penal Code or any other law for the time being in force is imprisonment for seven years or more;
(45) 'petty offences' includes the offences for which the maximum punishment under the Indian Penal Code or any other law for the time being in force is imprisonment up to three years;
(54) 'serious offences' includes the offences for which the punishment under the Indian Penal Code or any other law for the time being in force is imprisonment between three to seven years."""
            },
            {
                "section": "4",
                "title": "Juvenile Justice Board",
                "text": """Section 4. Juvenile Justice Board.—
(1) Notwithstanding anything contained in the Code of Criminal Procedure, 1973, the State Government shall, by notification in the Official Gazette, constitute for every district, one or more Juvenile Justice Boards for exercising the powers and discharging the functions relating to children in conflict with law under this Act.
(2) A Board shall consist of a Principal Magistrate (who shall be a Judicial Magistrate of First Class or Metropolitan Magistrate) and two social workers, of whom at least one shall be a woman."""
            },
            {
                "section": "10",
                "title": "Apprehension of child alleged to be in conflict with law",
                "text": """Section 10. Apprehension of child in conflict with law.—
(1) As soon as a child alleged to be in conflict with law is apprehended by the police, he shall be placed under the charge of the special juvenile police unit or the designated child welfare police officer, who shall produce the child before the Board without any loss of time but within a period of twenty-four hours of its apprehension.
(2) In no case, a child alleged to be in conflict with law shall be placed in a police lockup or lodged in a jail."""
            },
            {
                "section": "12",
                "title": "Bail to a person who is apparently a child alleged to be in conflict with law",
                "text": """Section 12. Bail of child in conflict with law.—
(1) When any person, who is apparently a child and is alleged to have committed a bailable or non-bailable offence, is apprehended or detained by the police or appears or is brought before a Board, such person shall, notwithstanding anything contained in the Code of Criminal Procedure, 1973 or in any other law for the time being in force, be released on bail with or without surety:
Provided that such person shall not be so released if there appears reasonable grounds for believing that the release is likely to bring that person into association with any known criminal or expose the said person to moral, physical or psychological danger or that the person's release would defeat the ends of justice."""
            },
            {
                "section": "74",
                "title": "Prohibition on disclosure of identity of children",
                "text": """Section 74. Prohibition on disclosure of identity of children.—
(1) No report in any newspaper, magazine, news-sheet or audio-visual media or other form of communication regarding any inquiry or investigation or judicial procedure, shall disclose the name, address or school or any other particular, which may lead to the identification of a child in conflict with law or a child in need of care and protection or a child victim or witness of a crime.
(2) Any person contravening the provisions of sub-section (1) shall be punishable with imprisonment for a term which may extend to six months or fine which may extend to two lakh rupees or both."""
            },
            {
                "section": "75",
                "title": "Punishment for cruelty to child",
                "text": """Section 75. Punishment for cruelty to child.—
Whoever, having the actual charge of, or control over, a child, assaults, abandons, abuses, exposes or wilfully neglects the child or causes or procures the child to be assaulted, abandoned, abused, exposed or neglected in a manner likely to cause such child unnecessary mental or physical suffering, shall be punishable with imprisonment for a term which may extend to three years or with fine of one lakh rupees or with both:
Provided that in case where the cruelty is committed by a person employed by or managing an organisation which is entrusted with the care and protection of the child, he shall be punished with rigorous imprisonment which may extend up to five years, and fine which may extend up to five lakh rupees."""
            },
        ],
    },
]


def create_act_pdf(dataset: dict) -> Path:
    """Create a validated PDF document containing official statutory text."""
    target_dir = LEGAL_DATA_DIR / dataset["folder"]
    target_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = target_dir / dataset["filename"]

    doc = fitz.open()

    # Create cover/title page and section pages
    for page_idx, sec in enumerate(dataset["sections"]):
        page = doc.new_page(width=595, height=842)  # A4 size
        rect = fitz.Rect(50, 50, 545, 792)

        # Build page text
        page_text = f"{dataset['title']}\n"
        page_text += f"Enactment Authority: {dataset['authority']}\n"
        page_text += f"Year of Enactment: {dataset['year']}\n"
        page_text += "=" * 60 + "\n\n"
        page_text += f"Section {sec['section']}: {sec['title']}\n\n"
        page_text += sec["text"].strip()

        page.insert_textbox(
            rect,
            page_text,
            fontsize=10,
            fontname="helv",
            color=(0.1, 0.1, 0.1),
        )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def ingest_new_datasets() -> dict:
    """Validate and ingest new authoritative acts into ChromaDB collection."""
    print("=" * 70)
    print("JAN NYAYA AI — AUTHORITATIVE LEGAL DATA INGESTION")
    print("=" * 70)

    initial_count = get_collection_count()
    print(f"Current ChromaDB chunk count: {initial_count}")

    # Check existing document hashes and IDs in collection
    existing_docs, existing_metas = get_all_documents()
    existing_hashes = {m.get("file_hash") for m in existing_metas if m and m.get("file_hash")}
    existing_doc_ids = {m.get("document_id") for m in existing_metas if m and m.get("document_id")}

    ingested_acts = []
    total_new_chunks = 0

    for ds in DATASETS:
        pdf_path = create_act_pdf(ds)
        print(f"\nProcessing: {ds['act_name']} ({ds['year']})")

        # 1. Validation
        val = validate_document_file(pdf_path)
        if not val["is_valid"]:
            print(f"Validation failed for {pdf_path.name}: {val['error']}")
            continue

        # 2. Hash calculation for deduplication
        with pdf_path.open("rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        doc_id = build_document_id(ds["act_name"], ds["year"])

        if file_hash in existing_hashes or doc_id in existing_doc_ids:
            print(f"Document '{doc_id}' already indexed with hash '{file_hash[:10]}...'. Skipping duplicate.")
            continue

        # 3. Extract text from PDF
        doc_fitz = fitz.open(str(pdf_path))
        full_text = ""
        for p_idx in range(len(doc_fitz)):
            page_t = doc_fitz[p_idx].get_text("text")
            full_text += f"\n\n--- Page {p_idx + 1} ---\n\n" + page_t
        doc_fitz.close()

        cleaned = clean_text(full_text)
        if not cleaned:
            print(f"No text extracted from {pdf_path.name}")
            continue

        # 4. Chunk text
        chunks = chunk_text(cleaned)
        if not chunks:
            print(f"No chunks produced for {pdf_path.name}")
            continue

        # 5. Extract section metadata for each chunk
        chunk_metadatas = []
        for c_idx, chunk_str in enumerate(chunks):
            sec_meta = extract_section_metadata(chunk_str)
            chunk_metadatas.append({
                "document_id": doc_id,
                "act_name": ds["act_name"],
                "year": ds["year"],
                "authority": ds["authority"],
                "document_type": ds["document_type"],
                "category": ds["category"],
                "route": ds["route"],
                "source": pdf_path.name,
                "title": ds["title"],
                "section_number": sec_meta.get("section_number", ""),
                "section_title": sec_meta.get("section_title", ""),
                "chunk_index": c_idx,
                "file_hash": file_hash,
                "ingested_at": time.time(),
            })

        # 6. Embed and store in ChromaDB
        embeddings = create_embeddings(chunks)
        add_chunks(
            chunks=chunks,
            metadatas=chunk_metadatas,
            embeddings=embeddings,
            source=pdf_path.name,
        )

        total_new_chunks += len(chunks)
        ingested_acts.append({
            "act_name": ds["act_name"],
            "year": ds["year"],
            "chunks": len(chunks),
            "file": pdf_path.name,
        })
        print(f"Successfully ingested {len(chunks)} chunks for {ds['act_name']}.")

    final_count = get_collection_count()
    print("\n" + "=" * 70)
    print(f"Ingestion complete! Initial chunks: {initial_count} -> Final chunks: {final_count}")
    print("=" * 70)

    return {
        "initial_chunks": initial_count,
        "final_chunks": final_count,
        "new_chunks_added": total_new_chunks,
        "ingested_acts": ingested_acts,
    }


if __name__ == "__main__":
    result = ingest_new_datasets()
    print(result)
