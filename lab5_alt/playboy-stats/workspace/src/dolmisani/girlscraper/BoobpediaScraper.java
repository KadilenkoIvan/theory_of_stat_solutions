package dolmisani.girlscraper;

import java.io.IOException;
import java.util.Random;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;

public class BoobpediaScraper {
	
	private String name;
	private Document document;
	
	private BoobpediaScraper(String name) {
		
		this.name = name;
		
		try {
			this.document = Jsoup.connect(buildUrl(name)).get();
		} catch (IOException e) {
			this.document = null;
		}
	}
	
	
	private String buildUrl(String name) {
		
		String url = String.format(
				"http://www.boobpedia.com/boobs/%s", 
				name.replaceAll("\\s", "_"));
		
		return url;
	}
	
	private String getField(String fieldName) {
		
		if (document == null) {
			return null;
		}
		
		Element e = document
				.select(String.format("td:contains(%s) + td", fieldName))
				.first();
		
		return (e != null) ? e.text() : "-";
	}
	
	
	public String getName() {
		
		return name;
	}
	
	
	public String getEthnicity() {
		
		return getField("Ethnicity");		
	}
	
	
	public String getBoobs() {
		
		return getField("Boobs");	
	}
	
	
	public String getEyeColor() {
		
		return getField("Eye color");
	}
	

	public String getHair() {
		
		return getField("Hair");
	}
	

	public String getBodyType() {
		
		return getField("Body");
	}
	
	
	public String getShown() {
		
		return getField("Shown");
	}
	
	
	public String getPubicHair() {
		
		return getField("Pubic");
	}
	
	@Override
	public String toString() {
		
		if (document == null) {
			return String.format("%20s NOT FOUND", name);
		} else {
		
			return String.format("%20s; %10s; %10s; %15s; %15s; %15s; %15s; %15s", 
					getName(), 
					getEthnicity(),
					getBoobs(),
					getEyeColor(),
					getHair(),
					getBodyType(),
					getPubicHair(),
					getShown());
		}
	}

	
	public static void main(String[] args) throws IOException {
		
		Random r = new Random();
		
		for (String name : new PlaymatesNames().getNames()) {
			System.out.println(new BoobpediaScraper(name));
			try {
				Thread.sleep(1000*(r.nextInt(30)+1));
			} catch (InterruptedException e) {

			}
		}
		
	}

}
