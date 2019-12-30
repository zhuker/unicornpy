import json
from typing import List, Tuple

import datagen


class UnicornDataset:
    @staticmethod
    def from_json(startups_jsonl: str, rounds_jsonl: str, industry_sort_order_json: str = None):
        startups = datagen._read_startups_jsonl(startups_jsonl)
        industries = []
        for name, startup in startups.items():
            industries.extend(startup['industries'])
        industries = sorted(list(set(industries)))

        rounds = datagen._read_rounds_jsonl(rounds_jsonl)
        ftypes = []
        for round in rounds:
            if 'fundingType' not in round:
                print("fundingType not found in ", round)
            ftypes.append(round.get('fundingType', ' x '))
        ftypes = sorted(list(set(ftypes)))
        print(ftypes)

        industry_sort_idxs = list(range(0, len(industries)))
        if industry_sort_order_json is not None:
            with(open(industry_sort_order_json, 'r')) as f:
                industry_sort_idxs = json.load(f)

        return UnicornDataset(ftypes, industries, industry_sort_idxs)

    @staticmethod
    def dataset1():
        UNIQ_INDUSTRIES = ["3D Printing", "3D Technology", "A/B Testing", "Accounting", "Ad Exchange", "Ad Network",
                           "Ad Retargeting", "Ad Server", "Ad Targeting", "Adult", "Advanced Materials",
                           "Adventure Travel",
                           "Advertising", "Advertising Platforms", "Advice", "Aerospace", "Affiliate Marketing",
                           "AgTech",
                           "Agriculture", "Air Transportation", "Alternative Medicine", "American Football",
                           "Amusement Park and Arcade", "Analytics", "Android", "Angel Investment", "Animal Feed",
                           "Animation",
                           "App Discovery", "App Marketing", "Application Performance Management",
                           "Application Specific Integrated Circuit (ASIC)", "Apps", "Aquaculture", "Architecture",
                           "Archiving Service", "Art", "Artificial Intelligence", "Asset Management", "Assisted Living",
                           "Assistive Technology", "Association", "Auctions", "Audio", "Augmented Reality",
                           "Auto Insurance",
                           "Automotive", "Autonomous Vehicles", "B2B", "B2C", "Baby", "Banking", "Basketball",
                           "Battery",
                           "Beauty", "Big Data", "Billing", "Biofuel", "Bioinformatics", "Biomass Energy", "Biometrics",
                           "Biopharma", "Biotechnology", "Bitcoin", "Blockchain", "Blogging Platforms",
                           "Brand Marketing",
                           "Brewing", "Broadcasting", "Browser Extensions", "Building Maintenance", "Building Material",
                           "Business Development", "Business Information Systems", "Business Intelligence",
                           "Business Travel",
                           "CAD", "CMS", "CRM", "Call Center", "Cannabis", "Car Sharing", "Career Planning", "Casino",
                           "Casual Games", "Celebrity", "Charity", "Charter Schools", "Chemical",
                           "Chemical Engineering",
                           "Child Care", "Children", "CivicTech", "Civil Engineering", "Classifieds", "Clean Energy",
                           "CleanTech", "Clinical Trials", "Cloud Computing", "Cloud Data Services",
                           "Cloud Infrastructure",
                           "Cloud Management", "Cloud Security", "Cloud Storage", "Coffee", "Collaboration",
                           "Collaborative Consumption", "College Recruiting", "Comics", "Commercial",
                           "Commercial Insurance",
                           "Commercial Lending", "Commercial Real Estate", "Communication Hardware",
                           "Communications Infrastructure", "Communities", "Compliance", "Computer", "Computer Vision",
                           "Concerts", "Construction", "Consulting", "Consumer", "Consumer Applications",
                           "Consumer Electronics", "Consumer Goods", "Consumer Lending", "Consumer Research",
                           "Consumer Reviews", "Consumer Software", "Contact Management", "Content", "Content Creators",
                           "Content Delivery Network", "Content Discovery", "Content Marketing", "Content Syndication",
                           "Continuing Education", "Cooking", "Corporate Training", "Cosmetic Surgery", "Cosmetics",
                           "Coupons",
                           "Courier Service", "Coworking", "Craft Beer", "Creative Agency", "Credit", "Credit Bureau",
                           "Credit Cards", "Crowdfunding", "Crowdsourcing", "Cryptocurrency", "Customer Service",
                           "Cyber Security", "Cycling", "DIY", "DRM", "DSP", "Data Center", "Data Center Automation",
                           "Data Integration", "Data Mining", "Data Storage", "Data Visualization", "Database",
                           "Dating",
                           "Debit Cards", "Delivery", "Delivery Service", "Dental", "Desktop Apps", "Developer APIs",
                           "Developer Platform", "Developer Tools", "Diabetes", "Dietary Supplements",
                           "Digital Entertainment",
                           "Digital Marketing", "Digital Media", "Digital Signage", "Direct Marketing", "Direct Sales",
                           "Document Management", "Document Preparation", "Domain Registrar", "Drone Management",
                           "Drones",
                           "E-Commerce", "E-Commerce Platforms", "E-Learning", "E-Signature", "EBooks", "EdTech",
                           "Ediscovery",
                           "Education", "Elder Care", "Elderly", "Electric Vehicle", "Electrical Distribution",
                           "Electronic Design Automation (EDA)", "Electronic Health Record (EHR)", "Electronics",
                           "Email",
                           "Email Marketing", "Embedded Software", "Embedded Systems", "Emergency Medicine",
                           "Emerging Markets",
                           "Employee Benefits", "Employment", "Energy", "Energy Efficiency", "Energy Management",
                           "Energy Storage", "Enterprise", "Enterprise Applications",
                           "Enterprise Resource Planning (ERP)",
                           "Enterprise Software", "Entertainment", "Environmental Consulting",
                           "Environmental Engineering",
                           "Ethereum", "Event Management", "Event Promotion", "Events", "Eyewear", "Facebook",
                           "Facial Recognition", "Facilities Support Services", "Facility Management", "Family",
                           "Fantasy Sports", "Farming", "Fashion", "Fast-Moving Consumer Goods", "Field Support",
                           "File Sharing", "Film", "Film Distribution", "Film Production", "FinTech", "Finance",
                           "Financial Exchanges", "Financial Services", "First Aid", "Fitness", "Flash Storage",
                           "Fleet Management", "Flowers", "Food Delivery", "Food Processing", "Food and Beverage",
                           "Franchise",
                           "Fraud Detection", "Freelance", "Freemium", "Freight Service", "Fuel", "Funding Platform",
                           "Funerals", "Furniture", "GPS", "GPU", "Gambling", "Gamification", "Gaming", "Genetics",
                           "Geospatial", "Gift", "Gift Card", "Golf", "Google Glass", "GovTech", "Government",
                           "Graphic Design",
                           "Green Building", "GreenTech", "Grocery", "Group Buying", "Guides", "Hardware",
                           "Health Care",
                           "Health Diagnostics", "Health Insurance", "Hedge Funds", "Higher Education", "Home Decor",
                           "Home Health Care", "Home Improvement", "Home Renovation", "Home Services",
                           "Home and Garden",
                           "Homeland Security", "Hospital", "Hospitality", "Hotel", "Housekeeping Service",
                           "Human Computer Interaction", "Human Resources", "Humanitarian", "ISP", "IT Infrastructure",
                           "IT Management", "IaaS", "Identity Management", "Image Recognition", "Impact Investing",
                           "In-Flight Entertainment", "Incubators", "Independent Music", "Indoor Positioning",
                           "Industrial",
                           "Industrial Automation", "Industrial Design", "Industrial Engineering",
                           "Industrial Manufacturing",
                           "Information Services", "Information Technology",
                           "Information and Communications Technology (ICT)",
                           "Infrastructure", "Innovation Management", "InsurTech", "Insurance", "Intellectual Property",
                           "Intelligent Systems", "Interior Design", "Internet", "Internet Radio", "Internet of Things",
                           "Intrusion Detection", "Jewelry", "Journalism", "Knowledge Management", "LGBT",
                           "Landscaping",
                           "Language Learning", "Laser", "Last Mile Transportation", "Laundry and Dry-cleaning",
                           "Law Enforcement", "Lead Generation", "Lead Management", "Leasing", "Legal", "Legal Tech",
                           "Leisure",
                           "Lending", "Life Insurance", "Life Science", "Lifestyle", "Lighting", "Linux", "Livestock",
                           "Local",
                           "Local Advertising", "Local Business", "Local Shopping", "Location Based Services",
                           "Logistics",
                           "Loyalty Programs", "MMO Games", "Machine Learning", "Machinery Manufacturing",
                           "Made to Order",
                           "Management Consulting", "Management Information Systems", "Manufacturing",
                           "Mapping Services",
                           "Marine Technology", "Marine Transportation", "Market Research", "Marketing",
                           "Marketing Automation",
                           "Marketplace", "Mechanical Design", "Mechanical Engineering", "Media and Entertainment",
                           "Medical",
                           "Medical Device", "Meeting Software", "Men's", "Messaging", "Micro Lending", "Military",
                           "Millennials", "Mineral", "Mining", "Mining Technology", "Mobile", "Mobile Advertising",
                           "Mobile Apps", "Mobile Devices", "Mobile Payments", "Motion Capture", "Music",
                           "Music Education",
                           "Music Label", "Music Streaming", "Music Venues", "Musical Instruments", "NFC",
                           "Nanotechnology",
                           "National Security", "Natural Language Processing", "Natural Resources", "Navigation",
                           "Network Hardware", "Network Security", "Neuroscience", "News", "Nightclubs", "Non Profit",
                           "Nursing and Residential Care", "Nutraceutical", "Nutrition", "Office Administration",
                           "Oil and Gas",
                           "Online Auctions", "Online Forums", "Online Games", "Online Portals", "Open Source",
                           "Operating Systems", "Optical Communication", "Organic", "Organic Food",
                           "Outdoor Advertising",
                           "Outdoors", "Outsourcing", "PC Games", "PaaS", "Packaging Services", "Parenting", "Parking",
                           "Parks",
                           "Payments", "Peer to Peer", "Penetration Testing", "Performing Arts", "Personal Branding",
                           "Personal Development", "Personal Finance", "Personal Health", "Personalization", "Pet",
                           "Pharmaceutical", "Photo Editing", "Photo Sharing", "Photography", "Physical Security",
                           "Podcast",
                           "Point of Sale", "Politics", "Ports and Harbors", "Power Grid", "Precious Metals",
                           "Predictive Analytics", "Presentation Software", "Presentations", "Price Comparison",
                           "Primary Education", "Printing", "Privacy", "Private Cloud", "Private Social Networking",
                           "Procurement", "Product Design", "Product Management", "Product Research", "Product Search",
                           "Productivity Tools", "Professional Networking", "Professional Services",
                           "Project Management",
                           "Property Development", "Property Insurance", "Property Management", "Psychology",
                           "Public Relations", "Public Safety", "Public Transportation", "Publishing", "Q&A",
                           "QR Codes",
                           "Quality Assurance", "Quantified Self", "RFID", "Racing", "Railroad", "Reading Apps",
                           "Real Estate",
                           "Real Estate Investment", "Real Time", "Recreation", "Recreational Vehicles", "Recruiting",
                           "Recycling", "Rehabilitation", "Renewable Energy", "Rental", "Rental Property", "Reputation",
                           "Reservations", "Residential", "Resorts", "Restaurants", "Retail", "Retail Technology",
                           "Retirement",
                           "Ride Sharing", "Risk Management", "Robotics", "SEM", "SEO", "SMS", "SNS", "STEM Education",
                           "SaaS",
                           "Sales", "Sales Automation", "Same Day Delivery", "Satellite Communication", "Scheduling",
                           "Search Engine", "Secondary Education", "Security", "Self-Storage", "Semantic Search",
                           "Semantic Web", "Semiconductor", "Sensor", "Service Industry", "Sharing Economy", "Shipping",
                           "Shoes", "Shopping", "Shopping Mall", "Simulation", "Skill Assessment",
                           "Small and Medium Businesses", "Smart Building", "Smart Cities", "Smart Home", "Snack Food",
                           "Soccer", "Social", "Social Bookmarking", "Social CRM", "Social Entrepreneurship",
                           "Social Impact",
                           "Social Media", "Social Media Advertising", "Social Media Management",
                           "Social Media Marketing",
                           "Social Network", "Social News", "Social Recruiting", "Social Shopping", "Software",
                           "Software Engineering", "Solar", "Space Travel", "Spam Filtering", "Speech Recognition",
                           "Sponsorship", "Sporting Goods", "Sports", "Staffing Agency", "Stock Exchanges",
                           "Subscription Service", "Supply Chain Management", "Surfing", "Sustainability", "TV",
                           "TV Production", "Task Management", "Taxi Service", "Tea", "Technical Support",
                           "Telecommunications",
                           "Tennis", "Test and Measurement", "Text Analytics", "Textbook", "Textiles", "Theatre",
                           "Therapeutics", "Ticketing", "Timber", "Tobacco", "Tourism", "Toys", "Trade Shows",
                           "Trading Platform", "Training", "Transaction Processing", "Translation Service",
                           "Transportation",
                           "Travel", "Travel Accommodations", "Travel Agency", "Tutoring", "Twitter", "UX Design",
                           "Unified Communications", "Universities", "Vacation Rental", "Vending and Concessions",
                           "Venture Capital", "Vertical Search", "Veterinary", "Video", "Video Advertising",
                           "Video Chat",
                           "Video Conferencing", "Video Editing", "Video Games", "Video Streaming", "Video on Demand",
                           "Virtual Assistant", "Virtual Currency", "Virtual Desktop", "Virtual Goods",
                           "Virtual Reality",
                           "Virtual Workforce", "Virtual World", "Virtualization", "Visual Search", "VoIP",
                           "Vocational Education", "Warehousing", "Waste Management", "Water", "Water Purification",
                           "Water Transportation", "Wealth Management", "Wearables", "Web Apps", "Web Browsers",
                           "Web Design",
                           "Web Development", "Web Hosting", "Wedding", "Wellness", "Wholesale", "Wind Energy",
                           "Wine And Spirits", "Wired Telecommunications", "Wireless", "Women's", "Young Adults",
                           "eSports",
                           "iOS", "mHealth", "—"]

        UNIQ_FUNDINGTYPES = ["Angel", "Convertible Note", "Corporate Round", "Debt Financing", "Equity Crowdfunding",
                             "Funding Round", "Grant", "Initial Coin Offering", "Non-equity Assistance",
                             "Post-IPO Debt",
                             "Post-IPO Equity", "Post-IPO Secondary", "Pre-Seed", "Private Equity", "Secondary Market",
                             "Seed",
                             "Series A", "Series B", "Series C", "Series D", "Series E", "Series F", "Series G",
                             "Series H",
                             "Series I", "Series J", "Venture - Series Unknown"
                             ]

        # see tsne1.py
        # industries sorted by context
        SORTED_INDUSTRIES_IDXS = [348, 284, 476, 323, 110, 322, 45, 416, 107, 210, 579, 82, 568, 625, 496, 557, 299,
                                  209, 472,
                                  473, 230, 229, 300, 345, 344, 620, 340, 157, 92, 273, 403, 274, 319, 419, 128, 256,
                                  317, 121,
                                  429, 236, 153, 79, 366, 318, 73, 365, 312, 72, 597, 78, 520, 217, 309, 341, 373, 589,
                                  487,
                                  401, 281, 200, 218, 74, 48, 49, 191, 540, 320, 466, 638, 216, 302, 215, 159, 164, 99,
                                  165,
                                  160, 627, 163, 100, 103, 459, 101, 642, 77, 185, 304, 422, 98, 464, 518, 523, 303,
                                  474, 431,
                                  576, 30, 183, 471, 468, 408, 116, 438, 102, 194, 154, 330, 463, 526, 305, 511, 450,
                                  563, 60,
                                  254, 3, 161, 117, 55, 56, 23, 559, 172, 457, 263, 162, 228, 37, 362, 583, 564, 486,
                                  404, 529,
                                  307, 70, 105, 237, 276, 614, 615, 629, 382, 389, 203, 515, 35, 204, 479, 271, 483, 32,
                                  333,
                                  595, 461, 139, 336, 525, 606, 602, 193, 441, 190, 195, 137, 630, 396, 550, 192, 584,
                                  286, 517,
                                  87, 560, 621, 25, 222, 636, 635, 208, 38, 259, 285, 64, 609, 152, 31, 63, 569, 594,
                                  242, 509,
                                  243, 241, 150, 51, 244, 383, 442, 111, 14, 347, 219, 126, 596, 167, 147, 149, 437,
                                  436, 393,
                                  255, 148, 539, 11, 360, 492, 502, 458, 184, 607, 475, 501, 491, 477, 504, 112, 343,
                                  144, 533,
                                  412, 358, 268, 399, 352, 129, 380, 452, 260, 371, 127, 532, 122, 28, 34, 608, 623,
                                  544, 123,
                                  287, 289, 253, 326, 290, 291, 41, 339, 297, 426, 252, 104, 67, 251, 506, 421, 250,
                                  278, 548,
                                  549, 418, 357, 182, 374, 42, 647, 628, 269, 537, 145, 570, 142, 279, 261, 270, 189,
                                  331, 558,
                                  643, 226, 536, 519, 249, 156, 535, 460, 507, 138, 428, 188, 508, 234, 566, 235, 350,
                                  590, 125,
                                  453, 54, 141, 381, 50, 440, 585, 433, 311, 86, 639, 272, 90, 493, 603, 169, 168, 521,
                                  631,
                                  571, 143, 359, 257, 534, 106, 248, 598, 19, 155, 81, 489, 510, 481, 338, 577, 327,
                                  296, 503,
                                  599, 295, 601, 591, 346, 280, 75, 600, 308, 435, 22, 655, 354, 524, 292, 650, 626,
                                  445, 9, 18,
                                  231, 17, 245, 197, 413, 196, 400, 611, 637, 233, 39, 288, 33, 176, 246, 469, 26, 644,
                                  443, 80,
                                  294, 58, 478, 207, 140, 414, 498, 425, 282, 170, 20, 353, 587, 175, 378, 61, 349, 446,
                                  62,
                                  267, 405, 578, 283, 89, 97, 409, 40, 201, 158, 314, 654, 379, 205, 582, 321, 293, 247,
                                  423,
                                  313, 485, 467, 640, 118, 306, 641, 604, 174, 44, 76, 298, 1, 173, 275, 444, 364, 265,
                                  505,
                                  430, 83, 21, 264, 617, 52, 146, 652, 266, 171, 361, 420, 84, 392, 91, 592, 391, 605,
                                  24, 653,
                                  516, 470, 334, 166, 227, 69, 130, 551, 465, 547, 555, 94, 490, 553, 448, 447, 449,
                                  151, 65,
                                  488, 484, 581, 545, 567, 411, 439, 232, 610, 119, 586, 225, 224, 27, 588, 223, 514,
                                  528, 390,
                                  542, 7, 5, 4, 29, 13, 8, 180, 342, 356, 12, 355, 6, 66, 372, 554, 16, 552, 178, 181,
                                  310, 427,
                                  513, 135, 134, 462, 2, 482, 332, 136, 179, 410, 556, 565, 494, 546, 651, 385, 377,
                                  132, 131,
                                  115, 238, 177, 397, 616, 239, 240, 394, 574, 575, 612, 593, 618, 619, 133, 613, 328,
                                  395, 68,
                                  398, 622, 451, 43, 645, 624, 36, 495, 108, 538, 500, 335, 368, 384, 46, 262, 406, 198,
                                  47,
                                  124, 202, 0, 367, 375, 512, 315, 572, 531, 15, 562, 325, 187, 186, 329, 351, 363, 543,
                                  417,
                                  541, 376, 199, 206, 120, 93, 71, 455, 213, 95, 211, 561, 221, 214, 646, 57, 633, 499,
                                  277,
                                  634, 212, 59, 96, 220, 573, 632, 497, 88, 258, 387, 386, 316, 456, 454, 388, 369, 337,
                                  370,
                                  480, 53, 402, 10, 109, 85, 432, 415, 434, 527, 522, 113, 114, 424, 407, 530, 580, 649,
                                  301,
                                  324, 648]

        return UnicornDataset(UNIQ_FUNDINGTYPES, UNIQ_INDUSTRIES, SORTED_INDUSTRIES_IDXS)

    def __init__(self, ftypes: List[str], uniq_industries: List[str], industry_sort_idxs: List[int]):
        self.ftypes = ftypes
        self.uniq_industries = uniq_industries
        self.industry_sort_idxs = industry_sort_idxs

    def shape(self) -> Tuple[int, int]:
        return len(self.ftypes), len(self.uniq_industries)

def read_json(jsonpath):
    with open(jsonpath, "r") as f:
        return json.load(f)

def read_uir(jsonpath):
    with open(jsonpath, "r") as f:
        useritemrating = json.load(f)

    ratings = useritemrating['ratings']
    funds = useritemrating['funds']
    startups = useritemrating['startups']
    startupIndustries = useritemrating.get('startupIndustries', {})
    industries = useritemrating.get('industries', {})
    return funds, startups, ratings, industries, startupIndustries


def fastratings(r):
    fast = {}
    for u, i, r in r:
        idx = u << 32 | i
        fast[idx] = r
    return fast


def getrating(fast, u, i):
    idx = u << 32 | i
    return fast.get(idx, 0)


if __name__ == '__main__':
    d2 = UnicornDataset.from_json('dataset2/startups2.jsonl', 'dataset2/rounds2.jsonl')
    print(d2.shape())
