package dolmisani.girlscraper;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class PlaymatesNames {
	
	private String[] names;
	
	public PlaymatesNames() {
		
		BufferedReader br = new BufferedReader(
				new InputStreamReader(
						this.getClass().getClassLoader().getResourceAsStream("names.txt")));
		
	
		List<String> l = new ArrayList<>();
		
		String name = null;
		try {
			while ((name = br.readLine()) != null) {
				l.add(name);
			}
		} catch (IOException e) {
		
		}
		
		names = l.toArray(new String[] { });
	}
	
		
	public String[] getNames() {
		return names;
	}
	
	public static void main(String[] args) {
		
		
		System.out.println(new PlaymatesNames().getNames().length);
	}
	
}
